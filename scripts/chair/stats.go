// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"runtime"
	"sort"
	"sync"
	"sync/atomic"
	"time"

	bolt "go.etcd.io/bbolt"
)

const maxScoreRows = 16384

type scoreRow struct {
	Worker     string  `json:"worker"`
	Model      string  `json:"model"`
	Completed  int64   `json:"completed"`
	Failed     int64   `json:"failed"`
	Canceled   int64   `json:"canceled"`
	Tokens     int64   `json:"tokens"`
	Measured   int64   `json:"measured"`
	RateTokens int64   `json:"rate_tokens"`
	RateMillis int64   `json:"rate_millis"`
	Samples    int64   `json:"samples"`
	Rate       float64 `json:"rate"`
}

type scoreMeta struct {
	Schema  int       `json:"schema"`
	Since   time.Time `json:"since"`
	Updated time.Time `json:"updated"`
	Missed  int64     `json:"missed"`
}

type scoreStore struct {
	db        *bolt.DB
	missed    atomic.Int64
	closeOnce sync.Once
	closeErr  error
}

type scoreView struct {
	Since     time.Time  `json:"since"`
	Updated   time.Time  `json:"updated"`
	Generated time.Time  `json:"generated"`
	Missed    int64      `json:"missed"`
	Model     string     `json:"model"`
	Models    []string   `json:"models"`
	Tokens    []scoreRow `json:"tokens"`
	Requests  []scoreRow `json:"requests"`
	Speed     []scoreRow `json:"speed"`
}

func openScoreStore(path string) (*scoreStore, error) {
	existed := false
	if before, err := os.Lstat(path); err == nil {
		existed = true
		if before.Size() == 0 {
			return nil, fmt.Errorf("existing statistics file is empty; choose a new path rather than overwriting it")
		}
		if !before.Mode().IsRegular() || (runtime.GOOS != "windows" && before.Mode().Perm()&0077 != 0) {
			return nil, fmt.Errorf("statistics file must be a regular owner-only file, not a symlink")
		}
	} else if !os.IsNotExist(err) {
		return nil, fmt.Errorf("cannot inspect statistics file")
	}
	db, err := bolt.Open(path, 0600, &bolt.Options{Timeout: time.Second})
	if err != nil {
		return nil, fmt.Errorf("cannot open statistics database; check permissions, file format, and whether another hub is using it")
	}
	s := &scoreStore{db: db}
	err = db.Update(func(tx *bolt.Tx) error {
		if existed && tx.Bucket([]byte("chair-scores-v1")) == nil {
			return fmt.Errorf("not a CHAIR statistics database")
		}
		bucket, err := tx.CreateBucketIfNotExists([]byte("chair-scores-v1"))
		if err != nil {
			return err
		}
		if bucket.Get([]byte("meta")) == nil {
			if existed {
				return fmt.Errorf("missing existing metadata")
			}
			if bucket.Stats().KeyN != 0 {
				return fmt.Errorf("missing metadata")
			}
			data, err := json.Marshal(scoreMeta{Schema: 1, Since: time.Now().UTC()})
			if err != nil {
				return err
			}
			return bucket.Put([]byte("meta"), data)
		}
		return nil
	})
	if err == nil {
		_, err = s.snapshot("")
	}
	if err != nil {
		_ = db.Close()
		return nil, fmt.Errorf("invalid or unreadable statistics database; do not overwrite it")
	}
	return s, nil
}

func (s *scoreStore) close() error {
	s.closeOnce.Do(func() { s.closeErr = s.db.Close() })
	return s.closeErr
}

func addScore(target *int64, value int64) error {
	if value < 0 || *target < 0 || *target > math.MaxInt64-value {
		return fmt.Errorf("statistics counter overflow")
	}
	*target += value
	return nil
}

func mergeScore(target *scoreRow, row scoreRow) error {
	for _, pair := range []struct {
		target *int64
		value  int64
	}{
		{&target.Completed, row.Completed}, {&target.Failed, row.Failed}, {&target.Canceled, row.Canceled}, {&target.Tokens, row.Tokens}, {&target.Measured, row.Measured}, {&target.RateTokens, row.RateTokens}, {&target.RateMillis, row.RateMillis}, {&target.Samples, row.Samples},
	} {
		if err := addScore(pair.target, pair.value); err != nil {
			return err
		}
	}
	return nil
}

func (s *scoreStore) record(worker, model, outcome string, usage *usageObserver, elapsed time.Duration) error {
	row := scoreRow{Worker: worker, Model: model}
	switch outcome {
	case "completed":
		row.Completed = 1
		if usage != nil && usage.eof && usage.terminal && !usage.failed && !usage.invalid && usage.tokens != nil {
			row.Tokens, row.Measured = *usage.tokens, 1
			if row.Tokens > 0 && elapsed.Milliseconds() > 0 {
				row.RateTokens, row.RateMillis, row.Samples = row.Tokens, elapsed.Milliseconds(), 1
			}
		}
	case "canceled":
		row.Canceled = 1
	default:
		row.Failed = 1
	}
	missed := s.missed.Swap(0)
	err := s.db.Update(func(tx *bolt.Tx) error {
		bucket := tx.Bucket([]byte("chair-scores-v1"))
		if bucket == nil {
			return fmt.Errorf("missing scores")
		}
		key := []byte("row:" + worker + "\x00" + model)
		var total scoreRow
		if data := bucket.Get(key); data != nil {
			if err := json.Unmarshal(data, &total); err != nil {
				return err
			}
		} else {
			if bucket.Stats().KeyN > maxScoreRows {
				return fmt.Errorf("statistics capacity exceeded")
			}
			total.Worker, total.Model = worker, model
		}
		if err := mergeScore(&total, row); err != nil {
			return err
		}
		data, err := json.Marshal(total)
		if err != nil {
			return err
		}
		if err := bucket.Put(key, data); err != nil {
			return err
		}
		var meta scoreMeta
		if err := json.Unmarshal(bucket.Get([]byte("meta")), &meta); err != nil {
			return err
		}
		if err := addScore(&meta.Missed, missed); err != nil {
			return err
		}
		meta.Updated = time.Now().UTC()
		data, err = json.Marshal(meta)
		if err != nil {
			return err
		}
		return bucket.Put([]byte("meta"), data)
	})
	if err != nil {
		s.missed.Add(missed + 1)
		return fmt.Errorf("cannot persist statistics")
	}
	return nil
}

func (s *scoreStore) snapshot(model string) (scoreView, error) {
	view := scoreView{Generated: time.Now().UTC(), Model: model, Models: []string{}, Tokens: []scoreRow{}, Requests: []scoreRow{}, Speed: []scoreRow{}}
	workers := make(map[string]scoreRow)
	models := make(map[string]bool)
	err := s.db.View(func(tx *bolt.Tx) error {
		bucket := tx.Bucket([]byte("chair-scores-v1"))
		if bucket == nil || bucket.Stats().KeyN > maxScoreRows+1 {
			return fmt.Errorf("invalid statistics bucket")
		}
		var meta scoreMeta
		if err := json.Unmarshal(bucket.Get([]byte("meta")), &meta); err != nil {
			return err
		}
		if meta.Schema != 1 || meta.Since.IsZero() || meta.Missed < 0 {
			return fmt.Errorf("invalid statistics version")
		}
		view.Since, view.Updated, view.Missed = meta.Since, meta.Updated, meta.Missed
		return bucket.ForEach(func(key, data []byte) error {
			if string(key) == "meta" {
				return nil
			}
			var row scoreRow
			if len(data) > 4096 || json.Unmarshal(data, &row) != nil || !validWorkerID(row.Worker) || !validModel(row.Model) || string(key) != "row:"+row.Worker+"\x00"+row.Model || row.Measured > row.Completed || row.Samples > row.Measured || row.RateTokens > row.Tokens {
				return fmt.Errorf("invalid statistics row")
			}
			var checked scoreRow
			if err := mergeScore(&checked, row); err != nil {
				return err
			}
			models[row.Model] = true
			if model != "" && row.Model != model {
				return nil
			}
			worker := workers[row.Worker]
			worker.Worker = row.Worker
			if err := mergeScore(&worker, row); err != nil {
				return err
			}
			workers[row.Worker] = worker
			if row.Samples >= 3 && row.RateTokens >= 32 && row.RateMillis > 0 {
				row.Rate = float64(row.RateTokens) * 1000 / float64(row.RateMillis)
				view.Speed = append(view.Speed, row)
			}
			return nil
		})
	})
	if err != nil {
		return view, fmt.Errorf("cannot read statistics")
	}
	if err := addScore(&view.Missed, s.missed.Load()); err != nil {
		return view, err
	}
	for name := range models {
		view.Models = append(view.Models, name)
	}
	sort.Strings(view.Models)
	for _, row := range workers {
		view.Requests = append(view.Requests, row)
		if row.Measured > 0 {
			view.Tokens = append(view.Tokens, row)
		}
	}
	tie := func(a, b scoreRow) bool {
		if a.Worker != b.Worker {
			return a.Worker < b.Worker
		}
		return a.Model < b.Model
	}
	sort.Slice(view.Tokens, func(i, j int) bool {
		a, b := view.Tokens[i], view.Tokens[j]
		if a.Tokens != b.Tokens {
			return a.Tokens > b.Tokens
		}
		return tie(a, b)
	})
	sort.Slice(view.Requests, func(i, j int) bool {
		a, b := view.Requests[i], view.Requests[j]
		if a.Completed != b.Completed {
			return a.Completed > b.Completed
		}
		return tie(a, b)
	})
	sort.Slice(view.Speed, func(i, j int) bool {
		a, b := view.Speed[i], view.Speed[j]
		if a.Rate != b.Rate {
			return a.Rate > b.Rate
		}
		return tie(a, b)
	})
	if len(view.Tokens) > 20 {
		view.Tokens = view.Tokens[:20]
	}
	if len(view.Requests) > 20 {
		view.Requests = view.Requests[:20]
	}
	if len(view.Speed) > 20 {
		view.Speed = view.Speed[:20]
	}
	return view, nil
}
