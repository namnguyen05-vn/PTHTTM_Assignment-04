# Thử nghiệm kiến trúc ban đầu

Phiên bản này có BatchNorm tại các convolution, residual block và dropout nhưng **chưa có BatchNorm sau Dense 64**. Bốn lượt thử đã chạy thật được lưu nguyên trạng, gồm cả log, checkpoint và test metrics.

Train/validation loss cao ở CIFAR-100 cho thấy underfitting. Phiên bản chính bổ sung BatchNorm sau Dense 64, trước ReLU; tất cả backend sau đó dùng kiến trúc cuối thống nhất. Pilot không thuộc bảng 18 cấu hình chính và không đủ để tách đóng góp riêng của từng cải tiến.

Để tái chạy pilot, từ thư mục này, đặt `CNN_DATA_DIR` đến dữ liệu đã chuẩn bị rồi chạy `python -m src.train --backend tensorflow --dataset cifar100 --variant improved --force`. Mã nguồn cũ được lưu trong `src/`; không thay thế mã chính bằng mã pilot.
