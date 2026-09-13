"""Synchronize the original six notebooks and add seed/ablation and worked-example notebooks."""
import runpy
from pathlib import Path
import nbformat as nbf

ROOT=Path(__file__).resolve().parents[1]
base=runpy.run_path(str(ROOT/'tools/build_notebooks.py'))
write,M,C=base['write'],base['M'],base['C']

# Keep the scope of the historical seed-42 notebooks clear in the extended project.
for path in (ROOT/'notebooks').glob('0[0-5]_*.ipynb'):
    nb=nbf.read(path,as_version=4)
    nb.cells.insert(1,M('**Bản mở rộng:** notebook này giữ nhóm 18 lượt seed 42 để giải thích thuật toán. '
                        'Notebook **06** tổng hợp đầy đủ 54 lượt nhiều seed và 18 ablation; '
                        'notebook **07** thực thi ví dụ số học dùng trong báo cáo 78 trang.'))
    for cell in nb.cells:
        if cell.cell_type=='code' and "'unittest','discover'" in cell.source:
            cell.source=cell.source.replace("[sys.executable,'-m','unittest','discover','-s','tests','-p','test_*.py','-v']",
                                            "[sys.executable,'tools/run_test_suite.py']")
        if cell.cell_type=='markdown' and 'Hướng tiếp theo là nhiều seed' in cell.source:
            cell.source=cell.source.replace('Hướng tiếp theo là nhiều seed, augmentation, lịch learning rate và ablation từng thành phần, tất cả lựa chọn bằng validation trước khi đánh giá test.',
                'Bản mở rộng đã bổ sung nhiều seed và ablation từng thành phần trong notebook 06. Augmentation và lịch learning rate vẫn là hướng chưa thực nghiệm. Nhận xét một seed ở đây chỉ mô tả nhóm cũ; kết luận tổng hợp cần đọc cùng notebook 06.')
    nbf.write(nb,path)

write('06_Multiple_Seeds_and_Ablation',[
M('''## 1. Giao thức cố định

**54 lượt chính = 3 dataset × 3 backend × 2 model × 3 seed.** Seed huấn luyện: 42, 7, 2026; split giữ seed 42. Thêm **18 ablation = 2 CIFAR × 3 can thiệp × 3 seed**, dùng PyTorch. Nhóm seed 42 cũ được tái sử dụng, không đếm lặp. Bốn pilot nằm ngoài 72 lượt.

`no_bn`: bỏ cả 4 BN; `no_skip`: chỉ bỏ phép cộng identity và giữ Conv/BN nhánh; `no_dropout`: p=0. Các can thiệp áp dụng sau khi nạp cùng trọng số khởi tạo improved. Chọn checkpoint theo validation loss, không chọn theo test. Test đã được quan sát trong quá trình phát triển nên đây là nghiên cứu khám phá.'''),
C("protocol=json.loads((ROOT/'results/extended_protocol.json').read_text())\nassert len(protocol['jobs'])==72\ndisplay(pd.DataFrame(protocol['jobs']).groupby(['backend','variant'],dropna=False).size().rename('runs'))\nprint('Seed huấn luyện:',protocol['seeds'],'| seed chia tập:',protocol['fixed_split_seed'])"),
M('''## 2. Mã điều phối và định nghĩa can thiệp

Mặc định đọc kết quả đã lưu. Đặt RUN_MISSING=True để chạy các lượt còn thiếu; runner tái sử dụng các lượt hoàn chỉnh và xác nhận cấu hình. Không tự ghi đè 72 lượt. Khi cần huấn luyện lại một lượt đã có, dùng CLI với --force và sao lưu trước.'''),
C("RUN_MISSING=False\nif RUN_MISSING:\n    subprocess.run([sys.executable,'run_extended.py'],cwd=ROOT,check=True)\nelse:\n    print('Đọc kết quả 72 lượt đã huấn luyện; không chạy lại.')"),
C("from src.torch_cnn import TorchCNN\nfrom src.numpy_cnn import CNN\nstate=CNN(3,32,100,'improved',7).state()\nmodels={}\nfor component in [None,'no_bn','no_skip','no_dropout']:\n    model=TorchCNN(3,32,100,'improved');model.load_numpy(state);model.ablate(component)\n    models[component or 'full']=sum(p.numel() for p in model.parameters())\ndisplay(pd.Series(models,name='trainable_parameters'))\nassert models=={'full':76020,'no_bn':75812,'no_skip':76020,'no_dropout':76020}"),
M('''## 3. Trung bình và độ lệch chuẩn

Dùng SD mẫu (ddof=1), không phải standard error và không phải khoảng tin cậy 95%. Ba seed trên cùng split chỉ đo một phần biến thiên của huấn luyện; không đánh giá biến thiên khi đổi nguồn ảnh. Không bỏ seed cho kết quả thấp.'''),
C("runs=pd.read_csv(ROOT/'results/extended_runs.csv')\nassert len(runs)==72 and runs.ablation.isna().sum()==54\nsummary=pd.read_csv(ROOT/'results/multiseed_summary.csv')\ndisplay(summary[['dataset','backend','variant','accuracy_mean','accuracy_std','macro_f1_mean','macro_f1_std','top5_accuracy_mean']])"),
C("for name in DATASETS:\n    display(Markdown('### '+name.upper()))\n    display(Image(filename=str(ROOT/'figures/extended'/f'{name}_mean_curves.png')))"),
M('''## 4. Chênh lệch ghép theo seed

Với mỗi dataset/backend, lấy accuracy improved trừ baseline của cùng seed rồi nhân 100 để được điểm phần trăm. Tính SD trực tiếp trên ba chênh lệch. Không dùng sự chồng lấn thanh SD làm phép kiểm định ý nghĩa thống kê.'''),
C("gains=pd.read_csv(ROOT/'results/paired_seed_gains.csv')\ndisplay(gains)\nassert len(gains)==9\nprint('Số cặp có improved tăng accuracy:',gains.positive_seeds.sum(),'/27')"),
M('''## 5. Ablation

Delta dưới đây là **biến thể trừ full improved**: âm nghĩa là tắt thành phần làm accuracy giảm. Đó là tác động có điều kiện trong kiến trúc và ngân sách hiện tại, chưa phải đóng góp độc lập áp dụng cho mọi CNN.'''),
C("ablation=pd.read_csv(ROOT/'results/ablation_summary.csv')\ndisplay(ablation)\nfor name in ['cifar10','cifar100']:\n    display(Image(filename=str(ROOT/'figures/extended'/f'{name}_ablation.png')))"),
M('''## 6. Trường hợp được sửa và bị làm sai thêm

Chọn trước PyTorch seed 42 cho cả ba dataset để minh họa. Dòng trên là ảnh baseline sai nhưng improved đúng; dòng dưới ngược lại. Lấy các test_id đầu tiên theo điều kiện, không chọn thủ công ảnh đẹp. T = nhãn thật; B = baseline; I = improved.'''),
C("for name in DATASETS:\n    detail=json.loads((ROOT/'results'/f'{name}_paired_errors.json').read_text())\n    print(detail)\n    assert sum(detail[k] for k in ['both_correct','corrected','regressed','both_wrong'])==10000\n    display(Image(filename=str(ROOT/'figures/extended'/f'{name}_paired_errors.png')))"),
M('''## 7. Trùng pixel và kiểm chứng đầu ra

Giữ test chính thức cho bảng chính. Phép phân tích độ nhạy CIFAR-100 loại 10 ảnh test đã xuất hiện bằng pixel trong training gốc, còn 9.990 ảnh. Nó không giải quyết near-duplicate hoặc trùng train-validation.'''),
C("display(pd.DataFrame(json.loads((ROOT/'results/extended_duplicate_sensitivity.json').read_text())))\nchecks=json.loads((ROOT/'results/extended_output_verification.json').read_text())\nassert len(checks)==72 and all(c['metrics_predictions_confusion_best_epoch_verified'] for c in checks)\nprint('Đã đối chiếu metric, dự đoán, confusion matrix và best epoch của 72 lượt.')"),
M('''## 8. Diễn giải

Trong các lượt đã chạy, improved tăng accuracy ở cả 27 cặp seed/dataset/backend. Bỏ BN làm giảm rõ trên hai bộ CIFAR. Tắt skip không làm chất lượng giảm nhất quán; CIFAR-10 còn tăng ở ba seed. Dropout có lợi rõ hơn trên CIFAR-100 trong ngân sách này. Các kết quả này không chứng minh mọi thành phần luôn cần thiết: mạng nhỏ, ba seed và chưa có mọi tổ hợp can thiệp.

Không chọn lại kiến trúc theo các kết quả test này rồi gọi cùng test là đánh giá độc lập. Chênh lệch nhỏ giữa framework chưa đủ để xếp hạng phổ quát; thời gian không phải benchmark cô lập.''')])

write('07_Worked_CNN_Example',[
M('''## 1. Ví dụ ảnh nhân tạo xuyên suốt

Ảnh 5×5, kernel 2×2, ReLU, MaxPool 2×2, Flatten, Dense 3 lớp, cross-entropy và một bước Adam. Đây là minh họa số học, không phải thí nghiệm trên MNIST/CIFAR. Mã bên dưới chính là module tạo các giá trị được đưa vào báo cáo.'''),
C((ROOT/'src/teaching_examples.py').read_text(encoding='utf-8').split("if __name__=='__main__':")[0].replace('from .numpy_cnn','from src.numpy_cnn').replace('from .data','from src.data')),
C("actual=example()\nreference=json.loads((ROOT/'results/teaching_example.json').read_text())\nfor key,value in actual.items():\n    if key!='purpose':np.testing.assert_allclose(value,reference[key],atol=1e-6,rtol=1e-6)\nprint('Toàn bộ giá trị khớp file dùng cho báo cáo.')"),
M('## 2. Convolution, phi tuyến và giảm không gian'),
C("for key in ['image','kernel','convolution','relu','pool','flatten']:\n    print(key);print(np.array(actual[key]))"),
M('## 3. Logits, xác suất và loss'),
C("for key in ['dense_weights','dense_bias','logits','probabilities','loss','logit_gradient']:\n    print(key);print(np.array(actual[key]))\nassert np.isclose(sum(actual['probabilities']),1)\nassert np.isclose(sum(actual['logit_gradient']),0,atol=1e-6)"),
M('''## 4. Gradient và cập nhật

Gradient kernel nhận tổng đóng góp từ các cửa sổ đã ảnh hưởng tới loss. Gradient input có thể bằng 0 ở vị trí bị ReLU hoặc pooling chặn. Adam cập nhật theo m/v, không đơn giản lấy learning rate nhân trực tiếp với gradient như SGD.'''),
C("for key in ['dense_gradient','kernel_gradient','input_gradient','kernel_after_adam']:\n    print(key);print(np.array(actual[key]))"),
M('''## 5. Các điểm cần giải thích khi bảo vệ

1. Tại sao convolution đầu ra là 4×4?
2. Vì sao ReLU làm một số gradient bằng 0?
3. Pooling trả gradient về đâu?
4. Vì sao gradient logits có tổng gần 0?
5. Vì sao cần cộng dồn trong col2im?
6. Một bước Adam khác một bước SGD như thế nào?

Đối chiếu chương 2 và 5 của báo cáo. Code đầy đủ của mạng huấn luyện nằm trong notebook 02; ví dụ nhỏ này chỉ hỗ trợ hiểu các phép tính.''')])
print('Synchronized 8 notebooks.')
