"""Build teaching notebooks; model/trainer cells are copied from canonical source."""
from pathlib import Path
import nbformat as nbf

ROOT=Path(__file__).resolve().parents[1]
DATASETS=['mnist','cifar10','cifar100']
OUT=ROOT/'notebooks';OUT.mkdir(exist_ok=True)
M=nbf.v4.new_markdown_cell;C=nbf.v4.new_code_cell
SETUP='''from pathlib import Path
import os, sys, json, subprocess
ROOT = Path.cwd().resolve()
if ROOT.name == 'notebooks': ROOT = ROOT.parent
assert (ROOT / 'src').is_dir(), 'Hãy mở notebook từ thư mục repository.'
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
from IPython.display import display, Markdown, Image
from src.data import DATASETS, load_data, data_root
print('Python:', sys.version.split()[0])
print('Dữ liệu:', data_root())'''

def write(name,cells):
    nb=nbf.v4.new_notebook(cells=[M('# '+name[3:].replace('_',' ')+'\n\nNguyễn Ngọc Hoàng Nam - B23DCCN585 | Assignment 04'),C(SETUP)]+cells)
    nb.metadata.kernelspec=dict(name='assignment04',display_name='Python (Assignment 04)',language='python')
    nb.metadata.language_info=dict(name='python',version='3.10.20')
    nbf.write(nb,OUT/(name+'.ipynb'))

write('00_Discover_CNN',[
M(r'''## 1. CNN là phép hợp thành hàm

Một mạng biến ảnh $x$ thành logits $z = f_L(\cdots f_2(f_1(x)))$. Softmax chuyển logits thành phân phối xác suất. Convolution học các bộ lọc cục bộ; ReLU tạo phi tuyến; pooling giảm kích thước; dense thực hiện phân loại. Nếu bỏ mọi phi tuyến giữa các phép biến đổi tuyến tính, các lớp tuyến tính liên tiếp vẫn có thể gộp thành một phép biến đổi tuyến tính.

Trọng số được chia sẻ giữa các vị trí ảnh, giúp giảm số tham số so với nối đầy đủ trực tiếp từ mọi pixel. CNN phù hợp với ảnh vì các pixel gần nhau mang thông tin không gian. Dịch chuyển không làm mạng bất biến tuyệt đối: padding, pooling và phần dense vẫn ảnh hưởng đáp ứng.'''),
M(r'''## 2. Convolution thủ công và kích thước

Các framework thường thực hiện **cross-correlation** (không lật kernel):

$$Y_{n,o,i,j}=b_o+\sum_{c,u,v}W_{o,c,u,v}X_{n,c,i+u-P,j+v-P}.$$

Với dilation $D=1$: $H_{out}=\lfloor(H+2P-K)/S\rfloor+1$. Số tham số convolution là $(K^2 C_{in}+1)C_{out}$. Trong bài, $K=3,P=1,S=1$ nên convolution giữ chiều rộng/cao. MaxPool 2×2 stride 2 giảm mỗi chiều một nửa.

Ví dụ 1D dưới đây tính đúng tổng tích; cần tự kiểm tra số học thay vì sao chép kết quả minh họa trong slide.'''),
C("x = np.array([2., 1., 3.]); kernel = np.array([.5, -1., .5])\nprint('Tổng tích =', x @ kernel)\nassert x @ kernel == 1.5"),
C('''from src.numpy_cnn import Conv2D
rng = np.random.default_rng(42)
layer = Conv2D(1, 1, rng, kernel=3, padding=0)
layer.w[:] = np.array([[[[-1,0,1],[-1,0,1],[-1,0,1]]]], dtype='float32')
layer.b[:] = 0
image = np.arange(25, dtype='float32').reshape(1,1,5,5)
print('Ảnh đầu vào:'); print(image[0,0])
print('Đáp ứng bộ lọc cạnh:'); print(layer.forward(image, False)[0,0])'''),
M(r'''## 3. Lan truyền ngược

Quy tắc dây chuyền đưa gradient từ loss về đầu vào. Với $G=\partial L/\partial Y$, gradient trọng số convolution cộng đóng góp ở mọi ảnh/vị trí. Mỗi pixel đầu vào nhận tổng gradient từ mọi cửa sổ chứa nó. Vì các cửa sổ chồng lấn, phép `col2im` phải **cộng dồn**, không ghi đè.

ReLU truyền gradient khi đầu vào dương; max pooling truyền về phần tử cực đại được chọn. Dense có $dW=X^T G$, $db=\sum G$, $dX=GW^T$. Cross-entropy kết hợp softmax có $\partial L/\partial z=(p-\text{onehot}(y))/N$.

Để ổn định số, trừ max của từng hàng logits trước khi tính exp. NumPy tự viết các đạo hàm và Adam; PyTorch dùng `backward`; TensorFlow dùng `GradientTape`.'''),
C('''from src.numpy_cnn import cross_entropy
logits = np.array([[1000.,1001.,999.]],dtype='float32')
loss, gradient = cross_entropy(logits,np.array([1]))
print('Loss hữu hạn:',loss,'; tổng gradient:',gradient.sum())'''),
M(r'''## 4. Cải tiến CNN

**BatchNorm:** $\hat{x}=(x-\mu)/\sqrt{\sigma^2+\epsilon}$, đầu ra $\gamma\hat{x}+\beta$. Trong train dùng thống kê batch; trong eval dùng running statistics. Bài này chuẩn hóa ở ba vị trí convolution và cả sau Dense 64, trước ReLU.

**Residual:** $y=\mathrm{ReLU}(F(x)+x)$, trong đó $F$ là Conv+BN. Hai nhánh cùng kích thước. Gradient được cộng qua nhánh identity và nhánh học được. Đây là block residual nhỏ cho bài thực hành, không phải toàn bộ ResNet-18.

**Dropout:** trong train giữ mỗi phần tử với xác suất 0,75 rồi chia cho 0,75; trong eval không che phần tử. Mục đích là regularization, không có bảo đảm tăng accuracy cho mọi bộ dữ liệu.

Bản thử đầu tiên chưa có BatchNorm sau Dense học chậm trên CIFAR-100. Phiên bản cuối thêm BN tại đây dựa trên loss train/validation, rồi dùng thống nhất ở cả ba backend. Kết quả bản thử được giữ tại `experiments/pilot_v1/`.'''),
C('''from src.numpy_cnn import CNN
rows=[]
for name,cfg in DATASETS.items():
    for variant in ['baseline','improved']:
        model=CNN(cfg['channels'],cfg['size'],cfg['classes'],variant)
        rows.append(dict(dataset=name,variant=variant,parameters=model.parameter_count()))
display(pd.DataFrame(rows))
model=CNN(3,32,100,'improved')
x=np.zeros((2,3,32,32),dtype='float32')
shapes=[]
for i,layer in enumerate(model.layers):
    x=layer.forward(x,False)
    shapes.append(dict(index=i,layer=type(layer).__name__,output_shape=str(x.shape)))
display(pd.DataFrame(shapes))'''),
M('''## 5. Thiết kế thí nghiệm và câu hỏi cần giải thích

Baseline có 2 Conv + 2 Dense; improved có 3 Conv + 2 Dense, cộng 4 BatchNorm. Luôn ghi rõ quy ước đếm lớp thay vì chỉ gọi “CNN nhiều lớp”.

Cùng split, preprocessing, initialization và minibatch order giúp so sánh kiến trúc/triển khai. Chọn checkpoint bằng validation loss; test phục vụ đánh giá cuối. Hãy giải thích: vì sao Dense chiếm nhiều tham số, vì sao 100 lớp khó hơn 10 lớp, vì sao dropout làm train accuracy khó so trực tiếp với eval accuracy, và vì sao cải tiến có thể cần thêm epoch.

Tham khảo: tài liệu CNN của học phần; Chollet (2021), chương đánh giá mô hình, CNN và kiến trúc hiện đại; [BN](https://arxiv.org/abs/1502.03167), [ResNet](https://arxiv.org/abs/1512.03385), [Dropout](https://www.jmlr.org/papers/v15/srivastava14a.html).''')])

write('01_Datasets',[
M('''## Nguồn dữ liệu

Ba bộ từ Kaggle: [MNIST](https://www.kaggle.com/datasets/oddrationale/mnist-in-csv), [CIFAR-10](https://www.kaggle.com/datasets/pankrzysiu/cifar10-python), [CIFAR-100](https://www.kaggle.com/datasets/fedesoriano/cifar100).

CIFAR-10 và CIFAR-100 đều có 60.000 ảnh màu 32×32. Chúng phức tạp hơn MNIST vì nền, màu, tư thế, đối tượng tự nhiên; CIFAR-100 có 100 lớp và ít ảnh mỗi lớp. “Big” được hiểu theo số lượng ảnh và độ phức tạp tương đối trong bài thực hành, không đồng nghĩa dữ liệu quy mô ImageNet.'''),
C('''missing=[name for name in DATASETS if not (data_root()/(name+'.npz')).exists()]
if missing:
    from src.download_data import download
    download()
else:
    print('Đã có đủ dữ liệu đã chuẩn bị; không tải lại.')
manifest=json.loads((ROOT/'results/dataset_manifest.json').read_text())
display(pd.DataFrame(manifest)[['dataset','train','validation','test','classes','shape','kaggle_url']])'''),
M('''## Chia tập và tiền xử lý

Giữ test chính thức. Tách validation 10% theo từng lớp từ training gốc, seed 42. MNIST: 54.001/5.999/10.000; CIFAR: 45.000/5.000/10.000. Không lấy mẫu con. Dữ liệu lưu uint8 để tiết kiệm RAM, chỉ đổi minibatch sang float32/255 khi dùng. Nhãn là số nguyên, loss sparse cross-entropy. Không fit bước tiền xử lý nào trên test.

File split và SHA-256 giúp kiểm tra hai backend có dùng đúng các mẫu giống nhau. Định dạng lưu chung NCHW; TensorFlow nhận NHWC sau transpose.'''),
C('''checks=[]
for name,cfg in DATASETS.items():
    d=load_data(name); ti,vi=d['train_ids'],d['val_ids']
    assert set(ti).isdisjoint(set(vi))
    assert len(ti)+len(vi)==len(d['y'])
    assert np.array_equal(np.sort(np.concatenate([ti,vi])),np.arange(len(d['y'])))
    assert d['x'].dtype==np.uint8
    checks.append(dict(dataset=name,train_shape=str(d['x'][ti[:1]].shape),pixel_min=int(d['x'].min()),pixel_max=int(d['x'].max()),all_indices_used=True))
    display(Image(filename=str(ROOT/'figures'/f'{name}_samples.png')))
display(pd.DataFrame(checks))'''),
C("audit=json.loads((ROOT/'results/dataset_audit.json').read_text())\ndisplay(pd.DataFrame(audit).drop(columns=['duplicate_policy']))"),
M('''## Kiểm tra trùng ảnh và giới hạn

Các chỉ số train/validation/test không giao nhau. Kiểm tra SHA-256 pixel không thấy ảnh trùng hoàn toàn giữa các phần MNIST/CIFAR-10. CIFAR-100 có 4 hash ảnh chung train-validation, 8 chung train-test và 2 chung validation-test. Giữ benchmark chính thức và công khai hạn chế này; không thay đổi dữ liệu theo kết quả test. Kiểm tra exact hash không phát hiện ảnh gần giống.

Vì có trùng pixel ở CIFAR-100, không khẳng định dữ liệu hoàn toàn không rò rỉ nội dung. Phần phân tích có thêm độ nhạy của accuracy khi loại các ảnh test đã xuất hiện trong training gốc; số này bổ sung cho metric benchmark chính thức.'''),
C("for name in DATASETS:\n    display(Image(filename=str(ROOT/'figures'/f'{name}_distribution.png')))"),
M('''## Dữ liệu đã chuẩn bị trước

Diabetes, VN housing và đánh giá quần áo có thể dành cho bài CNN 1D theo slide nếu được yêu cầu bổ sung. Dữ liệu bảng không có tính lân cận tự nhiên như ảnh; văn bản cần tokenize/embedding trước Conv1D. Chúng không được ghép vào ba thí nghiệm ảnh hiện tại.''')])

trainer=(ROOT/'src/train.py').read_text(encoding='utf-8').split("if __name__=='__main__':")[0]
trainer=trainer.replace('from .data','from src.data').replace('from .numpy_cnn','from src.numpy_cnn').replace('from .torch_cnn','from src.torch_cnn').replace('from .tf_cnn','from src.tf_cnn')
for index,backend,module,title in [(2,'numpy','numpy_cnn.py','CNN_From_Scratch_NumPy'),(3,'pytorch','torch_cnn.py','CNN_PyTorch'),(4,'tensorflow','tf_cnn.py','CNN_TensorFlow')]:
    intro={'numpy':'Toàn bộ convolution, pooling, dense, BatchNorm, residual, dropout, cross-entropy và Adam được viết bằng NumPy. Không dùng autograd hoặc lớp mạng của framework trong mô hình này.',
    'pytorch':'Mô hình dùng torch.nn và autograd. BatchNorm được viết bằng phép toán tensor để khớp phương sai tổng thể của bản NumPy. Huấn luyện dùng loss.backward() và torch.optim.Adam.',
    'tensorflow':'Mô hình dùng TensorFlow/Keras Functional API; vòng lặp dùng GradientTape và tf.function. Đầu vào NHWC, chuyển thứ tự kênh trước Flatten để cùng thứ tự đặc trưng với NumPy/PyTorch.'}[backend]
    cells=[M('## 1. Cách triển khai\n\n'+intro+'\n\nMã trong các ô dưới lấy trực tiếp từ module trong `src/`. Vòng lặp huấn luyện lưu checkpoint theo validation loss và chỉ đánh giá test sau khi khôi phục checkpoint tốt nhất.'),C("BACKEND = "+repr(backend)+"\nRETRAIN = False  # Đổi thành True để huấn luyện lại 6 cấu hình; sẽ ghi đè kết quả tương ứng.")]
    if backend=='tensorflow':
        cells.append(C("# Cấu hình GPU trước khi tạo bất kỳ tensor/model nào.\ncuda=os.environ.get('CNN_CUDA_DIR','E:/PTHTTM/ASG_04_runtime/cuda/Library/bin')\nif os.name=='nt' and Path(cuda).is_dir():\n    os.environ['PATH']=cuda+os.pathsep+os.environ['PATH']\n    dll_handle=os.add_dll_directory(cuda)\nimport tensorflow as tf\nfor gpu in tf.config.list_physical_devices('GPU'):\n    tf.config.experimental.set_memory_growth(gpu,True)\nprint('TensorFlow:',tf.__version__)"))
    cells += [M('## 2. Mã mô hình'),C((ROOT/'src'/module).read_text(encoding='utf-8')),M('## 3. Vòng lặp huấn luyện và đánh giá\n\nChương trình bên dưới chứa đầy đủ bước lấy batch, forward, loss, gradient, cập nhật, validation, checkpoint và đánh giá test. Các lượt chạy thực tế gọi cùng module trong một tiến trình riêng để cô lập framework.'),C(trainer),C('''def execute_dataset(name):
    rows=[]
    for variant in ['baseline','improved']:
        folder=ROOT/'results'/f'{name}_{BACKEND}_{variant}'
        if RETRAIN or not (folder/'metrics.json').exists():
            cmd=[sys.executable,'-m','src.train','--backend',BACKEND,'--dataset',name,'--variant',variant]
            if RETRAIN:cmd.append('--force')
            subprocess.run(cmd,cwd=ROOT,check=True)
        else:
            print('Đọc kết quả đã huấn luyện:',folder.name)
        rows.append(json.loads((folder/'metrics.json').read_text()))
        display(pd.read_csv(folder/'history.csv'))
    display(pd.DataFrame(rows)[['dataset','variant','accuracy','macro_f1','test_loss','top5_accuracy','best_epoch','train_seconds']])
    return rows''')]
    for name in DATASETS:
        cells += [M('## '+{'mnist':'4. MNIST','cifar10':'5. CIFAR-10','cifar100':'6. CIFAR-100'}[name]),C(f"{name}_results=execute_dataset('{name}')")]
    cells += [M('## 7. Kiểm tra triển khai'),C("verification=ROOT/'results'/f'verification_{BACKEND}.json'\nif verification.exists():\n    display(pd.DataFrame(json.loads(verification.read_text())))\nelse:\n    print(subprocess.run([sys.executable,'-m','unittest','discover','-s','tests','-p','test_*.py','-v'],cwd=ROOT,capture_output=True,text=True,check=True).stderr)"),M('''## Diễn giải

So sánh baseline với improved trong cùng dataset/backend. Accuracy không phản ánh toàn bộ chất lượng ở CIFAR-100: cần xem macro-F1, top-5 và các lớp hay nhầm. Train metrics được tích lũy trong lúc cập nhật trọng số, có dropout ở improved; validation chạy ở chế độ eval, nên không thể diễn giải mọi chênh lệch train-validation là overfitting.

Các chỉ số là một lượt chạy seed 42. Thời gian gồm bước huấn luyện thực tế nhưng không phải benchmark phần cứng độc lập. Notebook 05 đưa ra so sánh chung dựa trên 18 kết quả.''')]
    write(f'{index:02}_{title}',cells)

write('05_Compare_Results',[
M('''## 1. Đối chiếu đầu ra thực tế

Đọc logits và nhãn đã lưu, tính lại metric, kiểm tra confusion matrix và epoch được chọn. Không huấn luyện hoặc chọn lại mô hình bằng test trong bước này.'''),
C("from src.analyze import analyze\nsummary=analyze()\ndisplay(summary[['dataset','backend','variant','accuracy','macro_f1','top5_accuracy','test_loss','best_epoch','parameter_count','train_seconds']])"),
C("display(Image(filename=str(ROOT/'figures/accuracy_comparison.png')))"),
M('''## 2. Mức thay đổi khi cải tiến

Độ chênh dưới đây tính theo **điểm phần trăm**: accuracy improved trừ accuracy baseline. Không gọi chênh lệch này là phần trăm tăng tương đối.'''),
C("delta=summary.pivot(index=['dataset','backend'],columns='variant',values='accuracy')\ndelta['delta_percentage_points']=100*(delta.improved-delta.baseline)\ndisplay(delta)"),
M('''## 3. Learning curves và lỗi dự đoán

Mô hình minh họa mỗi dataset được chọn theo validation loss thấp nhất trong sáu lượt. Các ảnh lỗi là 10 dự đoán sai có confidence cao nhất; đây là tập lỗi có chủ đích để khảo sát, không đại diện cho tỷ lệ lỗi toàn test.'''),
C("for name in DATASETS:\n    display(Markdown('### '+name.upper()))\n    display(Image(filename=str(ROOT/'figures'/f'{name}_curves.png')))\n    display(Image(filename=str(ROOT/'figures'/f'{name}_confusion.png')))\n    display(Image(filename=str(ROOT/'figures'/f'{name}_errors.png')))\n    display(pd.DataFrame(json.loads((ROOT/'results'/f'{name}_error_analysis.json').read_text())['worst_pairs']))"),
M('''## 4. Độ nhạy với trùng ảnh CIFAR-100'''),
C("sensitivity=ROOT/'results/cifar100_duplicate_sensitivity.json'\nif sensitivity.exists():display(pd.DataFrame(json.loads(sensitivity.read_text())))"),
M('''## 5. Kết luận có thể rút ra

Mạng nhỏ đủ giải MNIST khá tốt nhưng ảnh tự nhiên, đặc biệt CIFAR-100, vẫn khó. Kết quả cải tiến cần đối chiếu với số tham số, số epoch và validation. So sánh một seed không đủ để khẳng định ưu thế có ý nghĩa thống kê giữa các framework.

Phiên bản improved thay nhiều thành phần cùng lúc, vì vậy thí nghiệm chính không tách được đóng góp riêng của BN, residual và dropout. Pilot chỉ khảo sát việc thêm BN ở phần dense; không thay thế ablation đầy đủ. Hướng tiếp theo là nhiều seed, augmentation, lịch learning rate và ablation từng thành phần, tất cả lựa chọn bằng validation trước khi đánh giá test.

Xem báo cáo PDF trong `report/` để đọc phân tích số liệu và giới hạn đầy đủ.''')])
print('Built',len(list(OUT.glob('*.ipynb'))),'notebooks')
