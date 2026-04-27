CC = gcc
CFLAGS = -Wall -Wextra -std=c99 -lm -I.
TARGETS = test_csv test_dataset test_iris test_binary test_gnb test_dt test_rf_n_ada test_ml test_softmax_benchmark test_idx test_tensor test_mlp test_lenet5 test_rnn test_lstm test_transformer main

.PHONY: all test test_dl clean

all: $(TARGETS)

test_csv: test/test_csv.c csv.h
	$(CC) $(CFLAGS) -o $@ test/test_csv.c

test_dataset: test/test_dataset.c csv.h dataset.h
	$(CC) $(CFLAGS) -o $@ test/test_dataset.c

test_iris: test/test_iris.c csv.h dataset.h machine_learning.h
	$(CC) $(CFLAGS) -o $@ test/test_iris.c

test_binary: test/test_binary.c csv.h dataset.h machine_learning.h
	$(CC) $(CFLAGS) -o $@ test/test_binary.c

main: main.c csv.h dataset.h machine_learning.h
	$(CC) $(CFLAGS) -o $@ main.c

test_ml: test/test_ml.c csv.h dataset.h ml_samples.h
	$(CC) $(CFLAGS) -o $@ test/test_ml.c

test_softmax_benchmark: test/test_softmax_benchmark.c csv.h dataset.h machine_learning.h linear_algebra.h mlr.h softmax_regression.h
	$(CC) $(CFLAGS) -o $@ test/test_softmax_benchmark.c

test_gnb: test/test_gnb.c csv.h dataset.h machine_learning.h gnb.h
	$(CC) $(CFLAGS) -o $@ test/test_gnb.c

test_dt: test/test_dt.c csv.h dataset.h machine_learning.h decision_tree.h
	$(CC) $(CFLAGS) -o $@ test/test_dt.c

test_idx: test/test_idx.c idx.h
	$(CC) $(CFLAGS) -o $@ test/test_idx.c

test_tensor: test/test_tensor.c tensor.h
	$(CC) $(CFLAGS) -o $@ test/test_tensor.c -lm

test_mlp: test/test_mlp.c mlp.h tensor.h idx.h
	$(CC) $(CFLAGS) -o $@ test/test_mlp.c -lm

test_rf_n_ada: test/rf_n_ada.c csv.h dataset.h machine_learning.h utilities.h adaboost.h randomforest.h
	$(CC) $(CFLAGS) -o $@ test/rf_n_ada.c -lm

# Deep Learning Tests
test_lenet5: test_lenet5.c lenet5.h tensor.h
	$(CC) $(CFLAGS) -o $@ test_lenet5.c -lm

test_rnn: test_rnn.c rnn.h tensor.h
	$(CC) $(CFLAGS) -o $@ test_rnn.c -lm

test_lstm: test_lstm.c lstm.h tensor.h
	$(CC) $(CFLAGS) -o $@ test_lstm.c -lm

test_transformer: test_transformer.c transformer.h tensor.h
	$(CC) $(CFLAGS) -o $@ test_transformer.c -lm

test_dl: test_lenet5 test_rnn test_lstm test_transformer
	@echo "=== Running LeNet-5 Tests ==="
	./test_lenet5
	@echo ""
	@echo "=== Running RNN Tests ==="
	./test_rnn
	@echo ""
	@echo "=== Running LSTM Tests ==="
	./test_lstm
	@echo ""
	@echo "=== Running Transformer Tests ==="
	./test_transformer

test: test_csv test_dataset test_iris test_binary test_gnb test_dt
	@echo "=== Running CSV Parser Tests ==="
	./test_csv
	@echo ""
	@echo "=== Running Dataset Tests ==="
	./test_dataset
	@echo ""
	@echo "=== Running Iris Tests ==="
	./test_iris data/iris.csv
	@echo ""
	@echo "=== Running Binary Classification Tests ==="
	./test_binary data/iris.csv
	@echo ""
	@echo "=== Running GNB Test ==="
	./test_gnb data/iris.csv
	@echo ""
	@echo "=== Running Decision Tree Test ==="
	./test_dt data/iris.csv

clean:
	rm -f $(TARGETS) test_softmax_benchmark test_softmax_benchmark_asan test_dl rf_n_ada
	rm -rf *.dSYM
