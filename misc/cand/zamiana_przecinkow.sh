#!/bin/bash

# Lista plików do przetworzenia
files=(
  "/Users/michal.dudzisz/Documents/mgr/flink-classifiers/results/jan-15_cand_atnn_like_mnist_do_wykresów/mnist_abrupt_atnn_like/cand/Psize30_Msize10/2025-01-16--00/part-2e2376ee-4158-44c3-9bd1-f0bbe8e54692-0"
  "/Users/michal.dudzisz/Documents/mgr/flink-classifiers/results/jan-15_cand_atnn_like_mnist_do_wykresów/mnist_abrupt_atnn_like/cand/Psize30_Msize10/2025-01-16--00/part-2e2376ee-4158-44c3-9bd1-f0bbe8e54692-1"
  "/Users/michal.dudzisz/Documents/mgr/flink-classifiers/results/jan-15_cand_atnn_like_mnist_do_wykresów/mnist_abrupt_atnn_like/cand/Psize30_Msize10/2025-01-16--00/part-2e2376ee-4158-44c3-9bd1-f0bbe8e54692-2"
  "/Users/michal.dudzisz/Documents/mgr/flink-classifiers/results/jan-15_cand_atnn_like_mnist_do_wykresów/mnist_abrupt_atnn_like/cand/Psize30_Msize10/2025-01-16--00/part-2e2376ee-4158-44c3-9bd1-f0bbe8e54692-3"
  "/Users/michal.dudzisz/Documents/mgr/flink-classifiers/results/jan-15_cand_atnn_like_mnist_do_wykresów/mnist_abrupt_atnn_like/cand/Psize30_Msize10/2025-01-16--00/part-2e2376ee-4158-44c3-9bd1-f0bbe8e54692-4"
  "/Users/michal.dudzisz/Documents/mgr/flink-classifiers/results/jan-15_cand_atnn_like_mnist_do_wykresów/mnist_abrupt_atnn_like/cand/Psize30_Msize10/2025-01-16--00/part-2e2376ee-4158-44c3-9bd1-f0bbe8e54692-5"
  "/Users/michal.dudzisz/Documents/mgr/flink-classifiers/results/jan-15_cand_atnn_like_mnist_do_wykresów/mnist_abrupt_atnn_like/cand/Psize30_Msize10/2025-01-16--00/part-2e2376ee-4158-44c3-9bd1-f0bbe8e54692-6"
  "/Users/michal.dudzisz/Documents/mgr/flink-classifiers/results/jan-15_cand_atnn_like_mnist_do_wykresów/mnist_abrupt_atnn_like/cand/Psize30_Msize10/2025-01-16--00/part-2e2376ee-4158-44c3-9bd1-f0bbe8e54692-7"
  "/Users/michal.dudzisz/Documents/mgr/flink-classifiers/results/jan-15_cand_atnn_like_mnist_do_wykresów/mnist_abrupt_atnn_like/cand/Psize30_Msize10/2025-01-16--00/part-2e2376ee-4158-44c3-9bd1-f0bbe8e54692-8"
  "/Users/michal.dudzisz/Documents/mgr/flink-classifiers/results/jan-15_cand_atnn_like_mnist_do_wykresów/mnist_abrupt_atnn_like/cand/Psize30_Msize10/2025-01-16--00/part-2e2376ee-4158-44c3-9bd1-f0bbe8e54692-9"
)

# Iteracja po plikach i wykonanie zamiany za pomocą sed
for file in "${files[@]}"; do
    if [ -f "$file" ]; then
        sed -i '' 's/GD_0,/GD_0./g; s/AM_0,/AM_0./g' "$file"
        echo "Zamiana zakończona w pliku: $file"
    else
        echo "Plik nie istnieje: $file"
    fi
done
