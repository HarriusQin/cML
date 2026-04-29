CC = gcc
CFLAGS = -Wall -Wextra -std=c99 -lm -I.
BINDIR = bin
TARGETS = $(BINDIR)/test_csv $(BINDIR)/test_dataset $(BINDIR)/test_iris $(BINDIR)/test_binary $(BINDIR)/test_gnb $(BINDIR)/test_dt $(BINDIR)/test_rf_n_ada $(BINDIR)/test_ml $(BINDIR)/test_softmax_benchmark $(BINDIR)/test_idx $(BINDIR)/test_tensor $(BINDIR)/test_mlp $(BINDIR)/test_lenet5 $(BINDIR)/test_rnn $(BINDIR)/test_lstm $(BINDIR)/test_transformer $(BINDIR)/test_dl_realdata_v2 $(BINDIR)/main

.PHONY: all test test_dl clean

all: $(TARGETS)

$(BINDIR):
	mkdir -p $(BINDIR)

$(BINDIR)/test_csv: test/test_csv.c csv.h | $(BINDIR)
	$(CC) $(CFLAGS) -o $@ test/test_csv.c

$(BINDIR)/test_dataset: test/test_dataset.c csv.h dataset.h | $(BINDIR)
	$(CC) $(CFLAGS) -o $@ test/test_dataset.c

$(BINDIR)/test_iris: test/test_iris.c csv.h dataset.h machine_learning.h | $(BINDIR)
	$(CC) $(CFLAGS) -o $@ test/test_iris.c

$(BINDIR)/test_binary: test/test_binary.c csv.h dataset.h machine_learning.h | $(BINDIR)
	$(CC) $(CFLAGS) -o $@ test/test_binary.c

$(BINDIR)/main: main.c csv.h dataset.h machine_learning.h | $(BINDIR)
	$(CC) $(CFLAGS) -o $@ main.c

$(BINDIR)/test_ml: test/test_ml.c csv.h dataset.h ml_samples.h | $(BINDIR)
	$(CC) $(CFLAGS) -o $@ test/test_ml.c

$(BINDIR)/test_softmax_benchmark: test/test_softmax_benchmark.c csv.h dataset.h machine_learning.h linear_algebra.h mlr.h softmax_regression.h | $(BINDIR)
	$(CC) $(CFLAGS) -o $@ test/test_softmax_benchmark.c

$(BINDIR)/test_gnb: test/test_gnb.c csv.h dataset.h machine_learning.h gnb.h | $(BINDIR)
	$(CC) $(CFLAGS) -o $@ test/test_gnb.c

$(BINDIR)/test_dt: test/test_dt.c csv.h dataset.h machine_learning.h decision_tree.h | $(BINDIR)
	$(CC) $(CFLAGS) -o $@ test/test_dt.c

$(BINDIR)/test_idx: test/test_idx.c idx.h | $(BINDIR)
	$(CC) $(CFLAGS) -o $@ test/test_idx.c

$(BINDIR)/test_tensor: test/test_tensor.c tensor.h | $(BINDIR)
	$(CC) $(CFLAGS) -o $@ test/test_tensor.c -lm

$(BINDIR)/test_mlp: test/test_mlp.c mlp.h tensor.h idx.h | $(BINDIR)
	$(CC) $(CFLAGS) -o $@ test/test_mlp.c -lm

$(BINDIR)/test_rf_n_ada: test/rf_n_ada.c csv.h dataset.h machine_learning.h utilities.h adaboost.h randomforest.h | $(BINDIR)
	$(CC) $(CFLAGS) -o $@ test/rf_n_ada.c -lm

# Deep Learning Tests
$(BINDIR)/test_lenet5: test_lenet5.c lenet5.h tensor.h | $(BINDIR)
	$(CC) $(CFLAGS) -o $@ test_lenet5.c -lm

$(BINDIR)/test_rnn: test_rnn.c rnn.h tensor.h | $(BINDIR)
	$(CC) $(CFLAGS) -o $@ test_rnn.c -lm

$(BINDIR)/test_lstm: test_lstm.c lstm.h tensor.h | $(BINDIR)
	$(CC) $(CFLAGS) -o $@ test_lstm.c -lm

$(BINDIR)/test_transformer: test_transformer.c transformer.h tensor.h | $(BINDIR)
	$(CC) $(CFLAGS) -o $@ test_transformer.c -lm

$(BINDIR)/test_dl_realdata_v2: test_dl_realdata_v2.c tensor.h idx.h rnn.h lstm.h transformer.h | $(BINDIR)
	$(CC) $(CFLAGS) -o $@ test_dl_realdata_v2.c -lm

test_dl: $(BINDIR)/test_lenet5 $(BINDIR)/test_rnn $(BINDIR)/test_lstm $(BINDIR)/test_transformer
	@echo "=== Running LeNet-5 Tests ==="
	./$(BINDIR)/test_lenet5
	@echo ""
	@echo "=== Running RNN Tests ==="
	./$(BINDIR)/test_rnn
	@echo ""
	@echo "=== Running LSTM Tests ==="
	./$(BINDIR)/test_lstm
	@echo ""
	@echo "=== Running Transformer Tests ==="
	./$(BINDIR)/test_transformer

test: $(BINDIR)/test_csv $(BINDIR)/test_dataset $(BINDIR)/test_iris $(BINDIR)/test_binary $(BINDIR)/test_gnb $(BINDIR)/test_dt
	@echo "=== Running CSV Parser Tests ==="
	./$(BINDIR)/test_csv
	@echo ""
	@echo "=== Running Dataset Tests ==="
	./$(BINDIR)/test_dataset
	@echo ""
	@echo "=== Running Iris Tests ==="
	./$(BINDIR)/test_iris data/iris.csv
	@echo ""
	@echo "=== Running Binary Classification Tests ==="
	./$(BINDIR)/test_binary data/iris.csv
	@echo ""
	@echo "=== Running GNB Test ==="
	./$(BINDIR)/test_gnb data/iris.csv
	@echo ""
	@echo "=== Running Decision Tree Test ==="
	./$(BINDIR)/test_dt data/iris.csv

clean:
	rm -rf $(BINDIR)
	rm -f test_softmax_benchmark_asan test_dl_realdata test_lenet5_bin test_mlp_asan test_mlp_iris
	rm -f test/test_mlp_framework test/test_mlp_iris test/test_transformer
	rm -f tests/test_tensor_edge_cases tests/test_tensor_edge_cases_asan
	rm -rf *.dSYM tests/*.dSYM
