// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"sync"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	bolt "go.etcd.io/bbolt"
)

func testScoreStore(t *testing.T) (*scoreStore, string) {
	t.Helper()
	path := filepath.Join(t.TempDir(), "scores.db")
	s, err := openScoreStore(path)
	require.NoError(t, err)
	t.Cleanup(func() { assert.NoError(t, s.close()) })
	return s, path
}

func measuredUsage(tokens int64) *usageObserver {
	return &usageObserver{eof: true, terminal: true, tokens: &tokens}
}

func TestScores_PersistTotalsAndSinceTimeAcrossReopen(t *testing.T) {
	s, path := testScoreStore(t)
	require.NoError(t, s.record("gpu-a", "coder", "completed", measuredUsage(42), time.Second))
	require.NoError(t, s.record("gpu-a", "coder", "completed", nil, time.Second))
	require.NoError(t, s.record("gpu-a", "coder", "failed", measuredUsage(100), time.Second))
	require.NoError(t, s.record("gpu-a", "coder", "canceled", measuredUsage(100), time.Second))
	before, err := s.snapshot("")
	require.NoError(t, err)
	require.NoError(t, s.close())
	reopened, err := openScoreStore(path)
	require.NoError(t, err)
	defer func() { assert.NoError(t, reopened.close()) }()
	after, err := reopened.snapshot("")
	require.NoError(t, err)
	assert.Equal(t, before.Since, after.Since)
	require.Len(t, after.Tokens, 1)
	assert.Equal(t, int64(42), after.Tokens[0].Tokens)
	assert.Equal(t, int64(1), after.Tokens[0].Measured)
	assert.Equal(t, int64(2), after.Requests[0].Completed)
	assert.Equal(t, int64(1), after.Requests[0].Failed)
	assert.Equal(t, int64(1), after.Requests[0].Canceled)
}

func TestScores_UseWeightedSpeedAndKeepModelsSeparate(t *testing.T) {
	s, _ := testScoreStore(t)
	for range 3 {
		require.NoError(t, s.record("gpu-a", "coder", "completed", measuredUsage(20), time.Second))
		require.NoError(t, s.record("gpu-b", "coder", "completed", measuredUsage(20), 2*time.Second))
	}
	require.NoError(t, s.record("gpu-a", "different", "completed", measuredUsage(999), time.Millisecond))
	view, err := s.snapshot("coder")
	require.NoError(t, err)
	require.Len(t, view.Speed, 2)
	assert.Equal(t, "gpu-a", view.Speed[0].Worker)
	assert.Equal(t, float64(20), view.Speed[0].Rate)
	assert.Equal(t, float64(10), view.Speed[1].Rate)
	assert.Equal(t, int64(60), view.Tokens[0].Tokens)
	assert.Equal(t, []string{"coder", "different"}, view.Models)
	all, err := s.snapshot("")
	require.NoError(t, err)
	assert.Len(t, all.Speed, 2, "a single tiny request cannot win the speed ranking")
	assert.Equal(t, int64(1059), all.Tokens[0].Tokens)
}

func TestScores_ConcurrentCompletionsDoNotLoseCredit(t *testing.T) {
	s, _ := testScoreStore(t)
	var group sync.WaitGroup
	for range 40 {
		group.Go(func() { assert.NoError(t, s.record("gpu-a", "coder", "completed", measuredUsage(10), time.Second)) })
	}
	group.Wait()
	view, err := s.snapshot("")
	require.NoError(t, err)
	require.Len(t, view.Tokens, 1)
	assert.Equal(t, int64(400), view.Tokens[0].Tokens)
	assert.Equal(t, int64(40), view.Requests[0].Completed)
}

func TestScores_RejectUnsafeFilesAndAnotherWriterWithoutOverwriting(t *testing.T) {
	s, path := testScoreStore(t)
	_, err := openScoreStore(path)
	assert.Error(t, err, "two hubs must not write the same statistics database")
	require.NoError(t, s.close())
	other := filepath.Join(t.TempDir(), "not-stats")
	require.NoError(t, os.WriteFile(other, []byte("PRIVATE_EXISTING_FILE"), 0600))
	_, err = openScoreStore(other)
	assert.Error(t, err)
	body, err := os.ReadFile(other)
	require.NoError(t, err)
	assert.Equal(t, "PRIVATE_EXISTING_FILE", string(body))
}

func TestScores_RejectForeignDatabaseWithoutAddingBuckets(t *testing.T) {
	path := filepath.Join(t.TempDir(), "foreign.db")
	db, err := bolt.Open(path, 0600, nil)
	require.NoError(t, err)
	require.NoError(t, db.Update(func(tx *bolt.Tx) error { _, err := tx.CreateBucket([]byte("foreign")); return err }))
	require.NoError(t, db.Close())
	_, err = openScoreStore(path)
	assert.Error(t, err)
	db, err = bolt.Open(path, 0600, nil)
	require.NoError(t, err)
	defer func() { assert.NoError(t, db.Close()) }()
	require.NoError(t, db.View(func(tx *bolt.Tx) error {
		assert.Nil(t, tx.Bucket([]byte("chair-scores-v1")))
		assert.NotNil(t, tx.Bucket([]byte("foreign")))
		return nil
	}))
}

func TestScores_KeepMissedUpdateWarningAfterStorageRecovers(t *testing.T) {
	s, path := testScoreStore(t)
	// Force a counter overflow to exercise rollback without platform-specific
	// disk permissions or modifying any actual filesystem capacity.
	require.NoError(t, s.db.Update(func(tx *bolt.Tx) error {
		data, err := json.Marshal(scoreRow{Worker: "gpu-a", Model: "coder", Completed: math.MaxInt64})
		if err != nil {
			return err
		}
		return tx.Bucket([]byte("chair-scores-v1")).Put([]byte("row:gpu-a\x00coder"), data)
	}))
	assert.Error(t, s.record("gpu-a", "coder", "completed", nil, time.Second))
	require.NoError(t, s.record("gpu-b", "coder", "completed", measuredUsage(12), time.Second))
	require.NoError(t, s.close())
	reopened, err := openScoreStore(path)
	require.NoError(t, err)
	defer func() { assert.NoError(t, reopened.close()) }()
	view, err := reopened.snapshot("")
	require.NoError(t, err)
	assert.Equal(t, int64(1), view.Missed)
}
