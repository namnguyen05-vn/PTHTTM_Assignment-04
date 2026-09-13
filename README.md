# Assignment 04 - CNN: NumPy, PyTorch và TensorFlow

**Nguyễn Ngọc Hoàng Nam - B23DCCN585 - D23CQCN11-B - Nhóm 05**  
Học phần Phát triển các hệ thống thông minh, Học viện Công nghệ Bưu chính Viễn thông.  
Giảng viên: **Trần Đình Quế**.

Bài làm theo yêu cầu trong ảnh: khám phá CNN, cải tiến CNN, cài đặt từ đầu và bằng PyTorch/TensorFlow trên **MNIST + CIFAR-10 + CIFAR-100**. Bản mở rộng gồm **báo cáo 78 trang, 8 notebook và 72 lượt thực nghiệm**: 54 lượt cho 18 cấu hình với 3 seed, cộng 18 ablation. CIFAR là hai bộ ảnh màu phức tạp hơn MNIST; cách hiểu “2 big” cần được giảng viên chấp nhận vì đề không nêu ngưỡng kích thước. Ba bộ dữ liệu dạng bảng/văn bản chuẩn bị trước được để ngoài phạm vi hiện tại.

## Bản mở rộng: nhiều seed và đánh giá thành phần

[Báo cáo PDF 78 trang](report/Assignment04_NguyenNgocHoangNam_B23DCCN585.pdf) trình bày mục tiêu nghiên cứu, cơ sở lý thuyết, phương pháp, ví dụ số học, 31 đoạn mã thuật toán và phân tích kết quả thực tế. Có **16 hình, 74 bảng**; phụ lục đặc tả điều kiện tái lập, còn lệnh cài đặt và thực thi nằm trong README. 8 notebook đã thực thi **60 ô code không lỗi**. Metric của 72 lượt đã được tính lại, 72 checkpoint đã được tải trong tiến trình CPU mới và đối chiếu sáu ảnh cố định; 15 unit test đạt.

**Accuracy improved, trung bình ± SD mẫu của 3 seed (%):**

| Dataset | NumPy | PyTorch | TensorFlow |
|---|---:|---:|---:|
| MNIST | 98.82 ± 0.10 | 98.84 ± 0.17 | 98.73 ± 0.06 |
| CIFAR-10 | 65.21 ± 1.06 | 65.21 ± 0.05 | 65.56 ± 0.66 |
| CIFAR-100 | 32.32 ± 0.16 | 32.41 ± 0.30 | 32.54 ± 0.14 |

Seed huấn luyện là **42, 7, 2026**; split giữ seed 42. Mười tám lượt cũ được tái sử dụng và chỉ đếm một lần. Dùng toàn bộ phần train, cùng batch 128, cùng ngân sách 5/10/12 epoch; không augmentation hoặc pretrained weights. `results/extended_protocol.json` mô tả 72 lượt. Bốn pilot ở `experiments/pilot_v1/` nằm ngoài tổng này.

**Ablation trên PyTorch, delta accuracy so với full improved (điểm phần trăm, trung bình ± SD của chênh lệch cùng seed):**

| Can thiệp | CIFAR-10 | CIFAR-100 |
|---|---:|---:|
| Bỏ cả 4 BN (`no_bn`) | -3.04 ± 0.55 | -6.25 ± 1.55 |
| Tắt riêng đường cộng (`no_skip`) | +0.40 ± 0.33 | +0.24 ± 0.80 |
| Tắt dropout (`no_dropout`) | -0.22 ± 0.58 | -0.98 ± 0.34 |

Can thiệp được áp dụng **sau khi nạp cùng trọng số khởi tạo improved**, rồi mỗi lượt học lại từ đầu. `no_skip` giữ nguyên convolution và BN của nhánh, nên số tham số không đổi. Kết quả cho thấy BN hữu ích trong cấu hình này; skip chưa mang lại lợi ích nhất quán. Improved tăng accuracy ở cả 27 cặp so với baseline, nhưng không phải mọi thành phần đều có lợi riêng lẻ. Đây là nghiên cứu khám phá trên một split và ba seed; SD không phải khoảng tin cậy, không đo biến thiên khi đổi dữ liệu, và thời gian không phải benchmark cô lập.

## Kết quả seed 42 ban đầu

Bảng dưới giữ riêng 18 lượt ban đầu để khớp các notebook thuật toán 02-05. Bảng tổng hợp cả ba seed nằm trong `results/multiseed_summary.csv` và notebook 06.

<!-- RESULTS_START -->
**Accuracy test (%): baseline → improved.** Mỗi kết quả là một lượt seed 42.

| Dataset | NumPy | PyTorch | TensorFlow |
|---|---:|---:|---:|
| MNIST | 98.69 → 98.92 | 98.63 → 99.03 | 98.68 → 98.75 |
| CIFAR10 | 63.15 → 66.43 | 63.03 → 65.26 | 63.01 → 66.02 |
| CIFAR100 | 28.23 → 32.42 | 28.84 → 32.15 | 28.54 → 32.54 |
<!-- RESULTS_END -->

## Dữ liệu và phạm vi thực nghiệm

| Dataset | Nguồn Kaggle | Train / validation / test | Ảnh | Lớp |
|---|---|---:|---|---:|
| MNIST | [MNIST in CSV](https://www.kaggle.com/datasets/oddrationale/mnist-in-csv) | 54.001 / 5.999 / 10.000 | 1 × 28 × 28 | 10 |
| CIFAR-10 | [CIFAR-10 Python](https://www.kaggle.com/datasets/pankrzysiu/cifar10-python) | 45.000 / 5.000 / 10.000 | 3 × 32 × 32 | 10 |
| CIFAR-100 | [CIFAR-100](https://www.kaggle.com/datasets/fedesoriano/cifar100) | 45.000 / 5.000 / 10.000 | 3 × 32 × 32 | 100 |

Validation lấy 10% theo từng lớp từ tập train chính thức, seed 42. MNIST có 5.999 mẫu validation do làm tròn riêng từng lớp. Dùng toàn bộ các phần dữ liệu đã chia, không lấy tập con. Test chính thức chỉ được đánh giá sau khi chọn checkpoint có validation cross-entropy thấp nhất. Chia pixel cho 255; không augmentation, không pretrained weights. Bảng ban đầu là kết quả seed 42; bảng mở rộng ở đầu README báo cáo trung bình và SD của ba seed huấn luyện.

## Cài đặt bằng Anaconda trên Windows

**Trên máy đã thực hiện bài:** kernel `Python (Assignment 04)` đã được đăng ký. Bạn có thể mở Jupyter từ Anaconda và chọn kernel này, hoặc chạy `START_JUPYTER.bat`. Python nằm ở `E:/PTHTTM/ASG_04_runtime/env/Scripts/python.exe`, dữ liệu ở `E:/PTHTTM/ASG_04_data`. Đây là môi trường venv riêng dựa trên Anaconda Python 3.10, kế thừa một số thư viện phân tích từ `intelsys_env`; không có thao tác thay đổi phiên bản thư viện trong môi trường Anaconda cũ. Phần dưới dành cho **tạo môi trường mới** trên máy khác hoặc tái lập sạch.

Mở **Anaconda Prompt** tại thư mục repository:

```bat
conda env create -f environment.yml
conda activate assignment04
python -m pip install torch==2.5.1+cu121 --index-url https://download.pytorch.org/whl/cu121
python -m ipykernel install --user --name assignment04 --display-name "Python (Assignment 04)"
set CNN_CUDA_DIR=%CONDA_PREFIX%\Library\bin
set CNN_DATA_DIR=E:\PTHTTM\ASG_04_data
python -m src.download_data
jupyter lab
```

Đổi `CNN_DATA_DIR` thành thư mục có đủ chỗ trên máy của bạn. Nếu không đặt biến, chương trình dùng `data/` trong repository, hoặc thư mục `E:/PTHTTM/ASG_04_data` nếu đã tồn tại. Cần khoảng 5 GB cho dữ liệu giải nén/cache và thêm dung lượng cho thư viện GPU. Dữ liệu gốc không được đưa vào Git vì kích thước lớn; đường dẫn và SHA-256 có trong `results/data_sources.json`.

TensorFlow **2.10.1** được ghim để dùng GPU trực tiếp trên Windows với CUDA 11.2/cuDNN 8.1. Đây là lựa chọn tương thích với máy thực nghiệm, không phải TensorFlow mới nhất. PyTorch dùng gói CUDA 12.1 riêng. Mỗi backend chạy trong một tiến trình riêng để tránh xung đột thư viện GPU. Máy không có GPU vẫn có thể chạy CPU nhưng chậm hơn. Không trộn hai framework trong cùng kernel đang dùng GPU.

Nếu Kaggle yêu cầu đăng nhập hoặc đổi API tải, tải thủ công ba ZIP từ các link trên, đặt tên lần lượt `mnist.zip`, `cifar10.zip`, `cifar100.zip` trong `CNN_DATA_DIR`, rồi chạy lại lệnh tải/chuẩn bị. Script sẽ tái sử dụng ZIP có sẵn.

## Các notebook

Chọn kernel **Python (Assignment 04)** và chạy theo thứ tự:

1. `notebooks/00_Discover_CNN.ipynb`: hợp thành hàm, convolution, kích thước, tham số, CNN cơ bản/cải tiến.
2. `notebooks/01_Datasets.ipynb`: nguồn Kaggle, cấu trúc dữ liệu, chia tập, ảnh minh họa, kiểm tra tính toàn vẹn.
3. `notebooks/02_CNN_From_Scratch_NumPy.ipynb`: mã forward/backward và Adam viết bằng NumPy, 6 cấu hình.
4. `notebooks/03_CNN_PyTorch.ipynb`: mô hình và huấn luyện PyTorch, 6 cấu hình.
5. `notebooks/04_CNN_TensorFlow.ipynb`: mô hình và huấn luyện TensorFlow/Keras, 6 cấu hình.
6. `notebooks/05_Compare_Results.ipynb`: bảng, learning curves, confusion matrix, phân tích lỗi và kết luận.
7. `notebooks/06_Multiple_Seeds_and_Ablation.ipynb`: 54 lượt nhiều seed, 18 ablation, chênh lệch ghép seed và phân tích lỗi bổ sung.
8. `notebooks/07_Worked_CNN_Example.ipynb`: ảnh nhân tạo 5×5 qua Conv, ReLU, Pool, Dense, cross-entropy, gradient và một bước Adam.

Notebook đã lưu các đầu ra được thực thi. Mặc định `RETRAIN = False` để đọc kết quả hoàn chỉnh nhanh. Đặt `RETRAIN = True` trong notebook thuật toán để huấn luyện lại **cả baseline và improved** của ba dataset. Các notebook hiển thị rõ khi đọc kết quả đã lưu; không giả lập log huấn luyện. Mã nguồn mô hình xuất hiện trực tiếp trong notebook và được quản lý tập trung ở `src/`.

Chạy giao thức mở rộng từ Anaconda Prompt (các lượt hoàn chỉnh sẽ được tái sử dụng):

```bat
python run_extended.py
python -m src.analyze_extended
python tools/verify_extended_checkpoints.py
```

Notebook 06 dùng `RUN_MISSING=False` mặc định. Đổi thành `True` để chạy các lượt còn thiếu. Các hàng đợi `--queue numpy_7`, `--queue numpy_2026` và `--queue gpu` cho phép chạy độc lập; hàng đợi GPU giữ các framework trong tiến trình riêng và chạy tuần tự. Không cần chạy đồng thời để tái lập.

Chạy nhóm seed 42 hoặc một lượt cụ thể:

```bat
python run_all.py
python run_all.py --backend numpy
python -m src.train --backend pytorch --dataset cifar100 --variant improved --force
python -m src.train --backend pytorch --dataset cifar100 --variant improved --seed 7 --ablation no_skip
```

Một cấu hình có `metrics.json` sẽ được bỏ qua, trừ khi có `--force`. Lượt bị ngắt chưa có metrics sẽ huấn luyện lại từ đầu. `--force` ghi đè kết quả cũ; sao lưu trước nếu muốn so sánh các lần chạy. Thay số epoch/batch size bằng tham số CLI sẽ tạo một thực nghiệm khác, không còn trực tiếp tương ứng bảng báo cáo.

## Mô hình và tính công bằng

- **Baseline:** Conv 3×3 (8 kênh) → ReLU → MaxPool → Conv 3×3 (16 kênh) → ReLU → MaxPool → Flatten → Dense 64 → ReLU → Dense C.
- **Improved:** thêm BatchNorm sau hai convolution, thêm residual block `ReLU(BN(Conv(x)) + x)` ở 16 kênh, BatchNorm sau Dense 64 trước ReLU và dropout 0,25 trước lớp đầu ra.
- Adam: learning rate 0,001; betas 0,9/0,999; epsilon 1e-8; batch size 128. MNIST 5 epoch, CIFAR-10 10 epoch, CIFAR-100 12 epoch.
- Cùng kiến trúc, split, thứ tự minibatch và bộ trọng số khởi tạo NumPy giữa các backend. TensorFlow dùng NHWC nhưng đổi về thứ tự NCHW trước Flatten.
- BatchNorm dùng phương sai tổng thể và running statistics tương đương. Lớp PyTorch được viết bằng phép toán tensor/autograd để tránh khác biệt mặc định về running variance.
- Kiểm tra số học tắt dropout để so sánh xác định. Huấn luyện thật bật dropout. Luồng ngẫu nhiên dropout, kernel GPU và chi tiết epsilon của Adam có thể khiến đường học khác nhau dù khởi tạo giống nhau.
- Thời gian ghi nhận trên máy dùng chung CPU/GPU trong quá trình hoàn thiện bài, không phải benchmark cô lập. Không suy ra framework nào nhanh hơn chỉ từ các con số này.

Bản thử đầu tiên chưa có BN ở phần Dense bị underfitting trên CIFAR-100 (train/validation loss cao). Phiên bản cuối bổ sung lớp này, thêm 128 tham số, và dùng thống nhất cho cả 3 backend. Bốn lượt thử cùng mã phiên bản cũ được giữ trong `experiments/pilot_v1/`; chúng không nằm trong 18 cấu hình chính. Quyết định điều chỉnh dựa trên train/validation, không dùng accuracy test để chọn cấu hình. Test của một số lượt thử cũng đã được ghi lại, nên đây là nghiên cứu khám phá, không phải đánh giá holdout hoàn toàn mù qua toàn bộ quá trình.

Kiểm tra exact pixel hash phát hiện CIFAR-100 có 4 hash chung train-validation, 8 chung train-test và 2 chung validation-test. Giữ nguyên benchmark và công khai hạn chế; `results/cifar100_duplicate_sensitivity.json` bổ sung accuracy trên các ảnh test chưa xuất hiện trong training gốc. Không phát hiện near-duplicate bằng kiểm tra hash.

## Đầu ra và kiểm tra

Mỗi thư mục `results/<dataset>_<backend>_<variant>/` chứa `config.json`, `history.csv`, `metrics.json`, `predictions.csv`, `confusion_matrix.csv`, `test_outputs.npz` và checkpoint tốt nhất (`weights.npz`, `.pt` hoặc `.h5`). `test_outputs.npz` lưu logits và nhãn để tính lại chỉ số; `test_id` là thứ tự mẫu trong tập test gốc. Checkpoint phục vụ suy luận, không chứa trạng thái optimizer để tiếp tục dở một epoch.

`results/summary.csv` tổng hợp 18 lượt seed 42. `results/extended_runs.csv` giữ đủ 72 lượt; `multiseed_summary.csv`, `paired_seed_gains.csv`, `ablation_summary.csv` tổng hợp thống kê. Seed mới được lưu trong `results/multiseed/seed_<s>/`, ablation trong `results/ablation/seed_<s>/`; không ghi đè seed 42. `figures/extended/` chứa đồ thị bổ sung. Báo cáo hoàn chỉnh nằm ở `report/Assignment04_NguyenNgocHoangNam_B23DCCN585.pdf`.

```bat
python -m unittest discover -s tests -p "test_*.py" -v
python -m tests.verify_framework --backend pytorch
python -m tests.verify_framework --backend tensorflow
python -m src.analyze
python tools/verify_extended_checkpoints.py
python -m src.predict --backend pytorch --dataset cifar10 --variant improved --test-id 123
```

Các phép kiểm tra sai phân hữu hạn xác nhận gradient convolution, dense, BatchNorm, pooling, residual và khả năng học bài toán nhỏ. Kiểm tra giữa các backend đối chiếu logits ở chế độ train/eval, gradient đầu vào và gradient convolution đầu tiên trên cả 6 kiến trúc/dataset. Đây là kiểm tra triển khai, không thay thế đánh giá test của mô hình đã huấn luyện.

Lệnh `src.predict` tải checkpoint trong tiến trình mới và hiển thị top-5 cho một ảnh test, tiện minh họa khi trình bày; thêm `--seed` và `--ablation` nếu cần. `verify_extended_checkpoints.py` đối chiếu sáu ảnh cố định của mỗi checkpoint với logits đã lưu, lưu 72 bản ghi trong `extended_checkpoint_verification.json`. Checksum của checkpoint, logits, config và mã suy luận giúp tự kiểm tra lại khi các file liên quan thay đổi. `results/runtime_versions.json` ghi phiên bản thực tế; `environment.yml` cung cấp bộ phiên bản tương thích để tạo môi trường sạch. Số luồng BLAS có thể khác giữa các lượt; không dùng thời gian trong bài làm benchmark.

Tái tạo bản đầy đủ sau khi có đủ 72 lượt:

```bat
python -m src.teaching_examples
python -m src.analyze
python -m src.analyze_extended
python tools/verify_extended_checkpoints.py
python tools/build_extended_notebooks.py
python tools/execute_notebooks.py
python report/build_report.py
python tools/verify_submission.py
```

Bộ tạo PDF dùng font Times New Roman và Consolas của Windows. Bước tạo notebook thay thế notebook cũ và cần thực thi lại để có đầu ra mới. `verify_submission.py` kiểm tra 72 lượt, 8 notebook, 60 ô code, 15 unit test và PDF 78 trang; kiểm tra logic này bổ sung cho việc đọc trực quan PDF. Raw dataset và sách/slide gốc không được đưa vào repository.

## Tài liệu tham khảo

- Slide học phần và *Deep Learning CNN Function Composition Tutorial* do giảng viên cung cấp.
- François Chollet, *Deep Learning with Python*, 2nd edition, 2021: đánh giá mô hình và workflow; CNN; kiến trúc hiện đại. Không đưa bản sách/slide có bản quyền vào repository.
- [Nguồn gốc CIFAR](https://www.cs.toronto.edu/~kriz/cifar.html).
- [Batch Normalization](https://arxiv.org/abs/1502.03167), [Deep Residual Learning](https://arxiv.org/abs/1512.03385), [Dropout](https://www.jmlr.org/papers/v15/srivastava14a.html).
- [TensorFlow: hướng dẫn cài đặt và giới hạn GPU Windows](https://www.tensorflow.org/install/pip).

Bài thực hành có dùng trợ lý AI hỗ trợ triển khai, kiểm tra và biên soạn. Các chỉ số trong báo cáo được tính từ lượt huấn luyện thực tế trên máy; người nộp cần hiểu được các bước và tuân thủ quy định sử dụng công cụ của học phần.
