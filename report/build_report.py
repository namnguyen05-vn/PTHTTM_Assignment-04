"""Generate the Vietnamese academic report from verified experiment outputs."""
import json,csv,html
from pathlib import Path
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_CENTER,TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.platypus import BaseDocTemplate,PageTemplate,Frame,Paragraph,Spacer,PageBreak,Table,TableStyle,Image,KeepTogether
from reportlab.platypus.tableofcontents import TableOfContents
from pypdf import PdfReader,PdfWriter

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'report';FIG=ROOT/'figures'
for name,file in [('TimesVN','times.ttf'),('TimesVN-Bold','timesbd.ttf'),('TimesVN-Italic','timesi.ttf'),('TimesVN-BoldItalic','timesbi.ttf')]:
    pdfmetrics.registerFont(TTFont(name,str(Path('C:/Windows/Fonts')/file)))
pdfmetrics.registerFontFamily('TimesVN',normal='TimesVN',bold='TimesVN-Bold',italic='TimesVN-Italic',boldItalic='TimesVN-BoldItalic')
INK=colors.HexColor('#182e42');BLUE=colors.HexColor('#214f70');PALE=colors.HexColor('#eef3f6');GRAY=colors.HexColor('#5b6269')
styles={
 'body':ParagraphStyle('Body',fontName='TimesVN',fontSize=12.2,leading=18,alignment=TA_JUSTIFY,spaceAfter=9),
 'h1':ParagraphStyle('Chapter',fontName='TimesVN-Bold',fontSize=17,leading=23,textColor=INK,spaceAfter=17),
 'h2':ParagraphStyle('Section',fontName='TimesVN-Bold',fontSize=15,leading=20,textColor=INK,spaceAfter=13),
 'sub':ParagraphStyle('Sub',fontName='TimesVN-Bold',fontSize=12.2,leading=17,spaceBefore=7,spaceAfter=7),
 'caption':ParagraphStyle('Caption',fontName='TimesVN-Italic',fontSize=10.3,leading=14,alignment=TA_CENTER,spaceBefore=5,spaceAfter=10),
 'cell':ParagraphStyle('Cell',fontName='TimesVN',fontSize=10.2,leading=13),
 'small':ParagraphStyle('Small',fontName='TimesVN',fontSize=10.3,leading=14,spaceAfter=7),
 'equation':ParagraphStyle('Equation',fontName='TimesVN',fontSize=12,leading=18,alignment=TA_CENTER,backColor=PALE,borderPadding=8,spaceBefore=4,spaceAfter=14)
}
class Report(BaseDocTemplate):
    def __init__(self,path):
        super().__init__(str(path),pagesize=A4,leftMargin=59,rightMargin=53,topMargin=60,bottomMargin=52,title='Assignment 04 - CNN - Nguyễn Ngọc Hoàng Nam',author='Nguyễn Ngọc Hoàng Nam')
        self.addPageTemplates(PageTemplate(id='main',frames=Frame(self.leftMargin,self.bottomMargin,self.width,self.height,id='content',leftPadding=0,rightPadding=0,topPadding=0,bottomPadding=0),onPage=self.decorate))
    def decorate(self,c,doc):
        c.saveState();c.setFont('TimesVN',9);c.setFillColor(GRAY)
        c.drawString(59,A4[1]-33,'PTIT  |  PHÁT TRIỂN CÁC HỆ THỐNG THÔNG MINH')
        c.drawRightString(A4[0]-53,A4[1]-33,'ASSIGNMENT 04')
        c.setStrokeColor(colors.HexColor('#bccbd5'));c.line(59,A4[1]-40,A4[0]-53,A4[1]-40)
        c.drawString(59,28,'Nguyễn Ngọc Hoàng Nam - B23DCCN585');c.drawRightString(A4[0]-53,28,str(doc.page));c.restoreState()
    def afterFlowable(self,f):
        if isinstance(f,Paragraph) and hasattr(f,'toc_level'):
            key='section_'+str(self.seq.nextf('section'));self.canv.bookmarkPage(key)
            self.notify('TOCEntry',(f.toc_level,f.getPlainText(),self.page,key))
            self.canv.addOutlineEntry(f.getPlainText(),key,level=f.toc_level,closed=False)

S=[];figure_no=0;table_no=0
def p(text,style='body'):S.append(Paragraph(text,styles[style]))
def page(title,chapter=False,toc=True):
    if S:S.append(PageBreak())
    para=Paragraph(title,styles['h1' if chapter else 'h2'])
    if toc:para.toc_level=0 if chapter else 1
    S.append(para)
def sub(text):p(text,'sub')
def eq(text):p(text,'equation')
def math_eq(lines):
    import io
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams['mathtext.fontset']='stix'
    plot=plt.figure(figsize=(6.7,.46*len(lines)+.2),facecolor='#eef3f6')
    for i,line in enumerate(lines):
        plot.text(.5,1-(i+.6)/(len(lines)+.25),'$'+line+'$',ha='center',va='center',fontsize=12,color='#182e42')
    buffer=io.BytesIO();plot.savefig(buffer,format='png',dpi=240,facecolor=plot.get_facecolor());plt.close(plot);buffer.seek(0)
    S.append(Image(buffer,width=483,height=(.46*len(lines)+.2)*72))
    S.append(Spacer(1,12))
def table(headers,rows,widths=None,caption=None):
    global table_no
    contents=[[Paragraph('<b>'+html.escape(str(v))+'</b>',styles['cell']) for v in headers]]
    contents += [[Paragraph(html.escape(str(v)),styles['cell']) for v in row] for row in rows]
    t=Table(contents,colWidths=widths,repeatRows=1,hAlign='LEFT')
    t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),PALE),('VALIGN',(0,0),(-1,-1),'TOP'),('LINEBELOW',(0,0),(-1,0),.7,BLUE),('LINEBELOW',(0,1),(-1,-1),.25,colors.HexColor('#d5dee4')),('LEFTPADDING',(0,0),(-1,-1),6),('RIGHTPADDING',(0,0),(-1,-1),6),('TOPPADDING',(0,0),(-1,-1),6),('BOTTOMPADDING',(0,0),(-1,-1),6)]))
    S.append(t)
    if caption:
        table_no+=1;p(f'Bảng {table_no}. {caption}','caption')
    else:S.append(Spacer(1,10))
def fig(name,caption,width=478):
    global figure_no
    im=Image(str(FIG/name));im.drawHeight*=width/im.drawWidth;im.drawWidth=width
    figure_no+=1;S.append(KeepTogether([im,Paragraph(f'Hình {figure_no}. '+caption,styles['caption'])]))
def link(url,label):return f'<a href="{html.escape(url,quote=True)}" color="#214f70">{html.escape(label)}</a>'
def pct(v):return f'{float(v)*100:.2f}'.replace('.',',')
def dec(v,n=4):return f'{float(v):.{n}f}'.replace('.',',')
def read(path):return json.loads((ROOT/path).read_text(encoding='utf-8'))
rows=list(csv.DictReader((ROOT/'results/summary.csv').open(encoding='utf-8')))
assert len(rows)==18,'Complete all 18 experiments before creating the final report.'
for r in rows:
    for k in ['accuracy','macro_precision','macro_recall','macro_f1','test_loss','top5_accuracy','best_val_loss','train_seconds','test_seconds']:r[k]=float(r[k])
    for k in ['parameter_count','best_epoch','epochs']:r[k]=int(r[k])
def get(name,backend,variant):return next(r for r in rows if r['dataset']==name and r['backend']==backend and r['variant']==variant)
def subset(name):return [r for r in rows if r['dataset']==name]
manifests=read('results/dataset_manifest.json');audits=read('results/dataset_audit.json')
gh='https://github.com/namnguyen05-vn/PTHTTM_Assignment-04'

page('LỜI MỞ ĐẦU',chapter=True,toc=False)
p('Báo cáo trình bày quá trình tìm hiểu mạng nơ-ron tích chập (CNN), xây dựng mô hình cơ bản và mô hình cải tiến, sau đó cài đặt bằng ba cách: NumPy từ đầu, PyTorch và TensorFlow. Mục tiêu là hiểu cách các phép biến đổi hợp thành một mô hình học được, kiểm chứng lan truyền ngược và thực hiện một quy trình đánh giá có thể tái lập.')
p('Phạm vi được chốt theo ảnh yêu cầu Assignment 04: ba dataset từ Kaggle gồm MNIST và hai bộ dữ liệu ảnh phức tạp hơn là CIFAR-10, CIFAR-100. Mỗi bộ có hai kiến trúc ở ba cách triển khai, tạo thành 18 thí nghiệm chính. Các bộ diabetes, nhà ở và đánh giá quần áo trong trao đổi ban đầu chưa thuộc phạm vi bài nộp này.')
p('Bài làm tham khảo slide học phần, tài liệu hướng dẫn CNN và các phần liên quan trong sách <i>Deep Learning with Python</i>, ấn bản thứ hai của François Chollet. Kết cấu báo cáo tham khảo mẫu Assignment 03 được cung cấp; số liệu và hình ảnh thực nghiệm được tạo từ chương trình của Assignment 04, không lấy từ báo cáo mẫu.')
table(['Nội dung cần thực hiện','Minh chứng trong bài'],[['Discover CNN','Lý thuyết, công thức, phép tính minh họa, kích thước và feature maps'],['Improved CNN','BatchNorm, residual, dropout; đối chiếu với baseline'],['From scratch / PyTorch / TensorFlow','Ba notebook thuật toán; mỗi notebook chạy hai mô hình trên ba dataset'],['Kết quả có thể kiểm tra','History, logits, nhãn, metric, confusion matrix và checkpoint'],['Bài nộp','Báo cáo PDF, notebook có đầu ra, README, mã nguồn GitHub']],[150,333],caption='Đối chiếu yêu cầu và thành phần bài nộp.')
p('Kho mã nguồn: '+link(gh,gh)+'. Các link Kaggle và hướng dẫn Anaconda có trong báo cáo và README. Bài thực hành có trợ lý AI hỗ trợ triển khai, kiểm tra và biên soạn; người nộp cần nắm được mã nguồn và các kết luận để trình bày, đồng thời tuân thủ quy định học phần.','small')

page('MỤC LỤC',toc=False)
toc=TableOfContents();toc.levelStyles=[ParagraphStyle('TOC0',fontName='TimesVN-Bold',fontSize=11,leading=13,spaceBefore=4),ParagraphStyle('TOC1',fontName='TimesVN',fontSize=10.5,leading=13,leftIndent=12,spaceBefore=0)]
S.append(toc)

page('CHƯƠNG 1. BÀI TOÁN VÀ THIẾT KẾ THỰC NGHIỆM',chapter=True)
sub('1.1. Mục tiêu và cách hiểu yêu cầu')
p('Đầu vào là ảnh và đầu ra là một nhãn thuộc C lớp. Mạng trả về C logits; lớp có logit lớn nhất là nhãn dự đoán. MNIST là bài toán chữ số 10 lớp; CIFAR-10 là ảnh tự nhiên 10 lớp; CIFAR-100 dùng 100 nhãn fine, không gộp thành 20 nhãn coarse.')
p('Đề không định nghĩa ngưỡng cho “2 big”. Bài này hiểu là hai dataset nhiều ảnh, phức tạp hơn MNIST, phù hợp khả năng huấn luyện từ đầu trên máy cá nhân. CIFAR-10 và CIFAR-100 đều có 60.000 ảnh màu, nhưng khác số lớp và mức phân biệt đối tượng. Cách chọn này đáp ứng hướng thực hành đã thống nhất, song chưa phải xác nhận của giảng viên rằng CIFAR được tính là “big”.')
p('Tính trên cả train gốc và test, mỗi bộ CIFAR chứa 184,32 triệu giá trị kênh-pixel, so với 54,88 triệu ở MNIST, tương đương khoảng 3,36 lần. Vì vậy ảnh ít hơn về số mẫu nhưng lớn hơn về lượng giá trị đầu vào và khó hơn về nội dung phân loại.','small')
sub('1.2. Câu hỏi nghiên cứu')
p('Thứ nhất, CNN viết bằng NumPy có gradient đúng và học được từ dữ liệu thực tế không? Thứ hai, cùng một kiến trúc có được biểu diễn tương đương trong NumPy, PyTorch và TensorFlow không? Thứ ba, việc thêm chuẩn hóa, residual và dropout làm thay đổi chất lượng phân loại ra sao ở ba mức độ khó?')
table(['Yếu tố','Thiết lập'],[['Dữ liệu','Toàn bộ train/validation/test đã định nghĩa; không subsampling'],['Hai mô hình','Baseline và improved cuối cùng, khởi tạo mới cho mỗi lượt'],['Ba triển khai','NumPy trên CPU; PyTorch và TensorFlow trên GPU'],['Chọn mô hình','Epoch có validation cross-entropy thấp nhất'],['Đánh giá','Accuracy, macro precision/recall/F1, top-5, loss, lỗi từng lớp']],[130,353],caption='Thiết kế chung của 18 cấu hình.')
p('Không dùng pretrained weights, transfer learning hoặc data augmentation trong thí nghiệm chính. Nhờ vậy, sự khác nhau giữa baseline và improved tập trung vào kiến trúc và regularization trong cùng ngân sách epoch; kết quả không được diễn giải là mức tốt nhất có thể đạt trên các benchmark này.')
p('Một tham chiếu đơn giản là luôn đoán lớp xuất hiện nhiều nhất trong phần train (chọn chỉ số nhỏ nhất khi hòa): accuracy test khi đó là 11,35% ở MNIST, 10% ở CIFAR-10 và 1% ở CIFAR-100. Các CNN được so sánh với nhau trong bảng chính, đồng thời cần vượt xa tham chiếu này để chứng tỏ đã học thông tin từ ảnh.','small')

page('1.3. Môi trường và các điều kiện so sánh')
table(['Thành phần','Môi trường thực nghiệm'],[['Hệ điều hành','Windows; Jupyter/Anaconda'],['CPU / RAM','Intel Core i5-12500H / 16 GB'],['GPU','NVIDIA GeForce RTX 3050 Laptop, 4 GB VRAM'],['Python / NumPy','Python 3.10.20; NumPy 1.23.5'],['PyTorch','2.5.1+cu121'],['TensorFlow','2.10.1; CUDA 11.2; cuDNN 8.1'],['Kernel notebook','Python (Assignment 04)']],[130,353],caption='Máy và môi trường chạy thực tế.')
p('Môi trường dùng Python 3.10 để tương thích TensorFlow 2.10.1. Theo hướng dẫn chính thức, 2.10 là phiên bản cuối hỗ trợ GPU trực tiếp trên Windows bản địa [12]. PyTorch dùng thư viện CUDA đi kèm gói cu121. Hai backend chạy trong tiến trình riêng để tránh xung đột DLL và trạng thái GPU.')
p('Tập chia, chuẩn hóa pixel, batch size, thứ tự minibatch theo epoch và bộ trọng số khởi tạo được dùng chung. Trọng số NumPy được chuyển đúng thứ tự sang các framework; số tham số trainable được kiểm tra bằng assert. Các kernel GPU vẫn có sai khác số thực, dropout dùng luồng ngẫu nhiên riêng và Adam có chi tiết triển khai khác nhau.')
table(['Siêu tham số','Giá trị'],[['Seed / batch size','42 / 128'],['Optimizer','Adam; learning rate 0,001; beta1 0,9; beta2 0,999; epsilon 1e-8'],['Epoch MNIST / CIFAR-10 / CIFAR-100','5 / 10 / 12'],['Chuẩn hóa','Pixel uint8 chuyển float32 và chia 255'],['Dropout improved','0,25; chỉ bật khi train'],['BatchNorm','Epsilon 1e-5; running = 0,9 × cũ + 0,1 × batch']],[185,298],caption='Siêu tham số chung, không dò theo test.')
p('Thời gian được đo trên máy đang đồng thời hoàn thiện các phần của bài. NumPy sử dụng CPU, hai framework sử dụng GPU; vì vậy số giây chỉ mô tả chi phí thực tế, không phải benchmark cô lập và không đủ kết luận framework nào nhanh hơn.','small')

page('CHƯƠNG 2. CƠ SỞ LÝ THUYẾT CNN',chapter=True)
sub('2.1. Mạng học sâu như phép hợp thành hàm')
p('Một mạng nơ-ron nhận tensor đầu vào và lần lượt áp dụng các hàm có tham số. Với CNN, chuỗi hàm thường gồm convolution, kích hoạt, pooling và lớp phân loại. Toàn bộ chuỗi tạo ra một hàm khả vi gần khắp nơi để tối ưu bằng gradient. Góc nhìn này kết nối ví dụ hợp thành hàm trong tài liệu hướng dẫn với mã forward/backward [1, 2].')
eq('z = f<sub>L</sub>(f<sub>L-1</sub>(... f<sub>2</sub>(f<sub>1</sub>(x; θ<sub>1</sub>); θ<sub>2</sub>) ... ); θ<sub>L</sub>)')
p('Convolution khai thác tính cục bộ: một bộ lọc nhỏ nhìn vào vùng ảnh gần nhau để học cạnh, nét hoặc mẫu màu. Cùng bộ lọc được dùng ở nhiều vị trí, giảm số tham số so với kết nối đầy đủ đến từng vị trí độc lập. Các tầng sau kết hợp đặc trưng từ tầng trước thành biểu diễn phù hợp nhãn cần dự đoán [3].')
p('ReLU đặt các giá trị âm về 0. Nếu mọi tầng chỉ gồm các phép tuyến tính hoặc affine, nhiều tầng liên tiếp vẫn có thể gộp thành một phép affine; độ sâu khi đó không tạo ra khả năng biểu diễn phi tuyến cần thiết. Pooling giảm kích thước không gian, giúp phần sau gọn hơn nhưng cũng làm mất thông tin vị trí chi tiết.')
fig('mnist_feature_maps.png','Ảnh test MNIST và tám feature map sau Conv1 + ReLU của CNN NumPy baseline đã học. Mỗi bản đồ được tô màu theo thang riêng, dùng để xem mẫu đáp ứng chứ không so biên độ giữa kênh.')
p('Các feature map thể hiện vùng mà từng bộ lọc đáp ứng. Không nên gán chắc chắn “bộ lọc này nhận biết chữ số” chỉ từ một ảnh minh họa. Tính hữu ích của đặc trưng cuối cùng được kiểm tra bằng chất lượng trên dữ liệu chưa dùng để cập nhật trọng số.')

page('2.2. Convolution, pooling và số tham số')
p('Trong mã và phần lớn framework, phép toán gọi là convolution thực hiện cross-correlation: kernel không được lật. Đầu ra ở một vị trí là tổng tích giữa kernel và cửa sổ đầu vào, cộng bias. NCHW lần lượt là số ảnh, số kênh, chiều cao và chiều rộng.')
eq('Y[n,o,i,j] = b[o] + Σ<sub>c,u,v</sub> W[o,c,u,v] X[n,c,i+u-P,j+v-P]')
eq('H<sub>out</sub> = floor((H + 2P - D(K-1) - 1)/S) + 1')
p('Với kernel 3×3, padding 1, stride 1 và dilation 1, convolution giữ nguyên chiều rộng/cao. MaxPool 2×2, stride 2 đưa ảnh 28×28 về 14×14 rồi 7×7; ảnh 32×32 về 16×16 rồi 8×8. MaxPool không có tham số cần học.')
table(['Lớp','Công thức tham số','Ví dụ'],[['Conv2D có bias','(K² × Cin + 1) × Cout','Conv 3→8, K=3: 224'],['Dense có bias','(Din + 1) × Dout','Dense 1.024→64: 65.600'],['BatchNorm','2 × số kênh/đặc trưng','64 đặc trưng: 128 trainable'],['ReLU / MaxPool / Dropout','0','Không có trọng số học']],[125,190,168],caption='Quy tắc đếm tham số trainable.')
p('Running mean và running variance của BatchNorm là trạng thái cập nhật theo dữ liệu, không phải tham số tối ưu bằng gradient nên không được tính vào cột trainable. Dense chiếm phần lớn tham số trong mạng nhỏ này do Flatten giữ mọi vị trí không gian.')
sub('Ví dụ số học độc lập')
eq('[2, 1, 3] · [0,5; -1; 0,5] = 2 × 0,5 - 1 + 3 × 0,5 = 1,5')
p('Notebook Discover thực thi ví dụ trên và assert kết quả. Việc kiểm tra phép tính nhỏ giúp phát hiện sai sót trong minh họa trước khi dùng công thức cho tensor lớn.')

page('2.3. Loss, lan truyền ngược và Adam')
p('Mạng xuất logits z. Softmax tạo xác suất p; sparse cross-entropy dùng nhãn nguyên thay vì lưu one-hot. Để tránh tràn exp, trừ logit lớn nhất trong từng mẫu trước khi tính softmax và log-sum-exp.')
math_eq([r'p_{i,c}=\frac{\exp(z_{i,c}-\max z_i)}{\sum_k \exp(z_{i,k}-\max z_i)}',
         r'L=-\frac{1}{N}\sum_i\log p_{i,y_i},\qquad \frac{\partial L}{\partial z}=\frac{p-\mathrm{onehot}(y)}{N}'])
p('Quy tắc dây chuyền cho phép truyền đạo hàm từ loss về từng lớp. Dense với Y=XW+b có dW=XᵀG, db là tổng G theo batch và dX=GWᵀ. ReLU giữ gradient tại đầu vào dương. MaxPool chuyển gradient đến vị trí cực đại đã lưu trong forward.')
p('Với convolution, dW cộng đóng góp ở mọi vị trí và mọi mẫu. Gradient về đầu vào phải cộng dồn từ các cửa sổ chồng lấn. Bản NumPy lưu ma trận im2col ở forward, nhân ma trận ở backward và dùng vòng lặp theo vị trí kernel để col2im. Thiếu phép cộng dồn sẽ cho gradient sai ngay cả khi kích thước tensor vẫn hợp lệ.')
sub('Cập nhật Adam')
math_eq([r'm_t=\beta_1m_{t-1}+(1-\beta_1)g_t',
         r'v_t=\beta_2v_{t-1}+(1-\beta_2)g_t^2',
         r'\hat{m}_t=m_t/(1-\beta_1^t),\qquad\hat{v}_t=v_t/(1-\beta_2^t)',
         r'\theta_t=\theta_{t-1}-\alpha\hat{m}_t/(\sqrt{\hat{v}_t}+\epsilon)'])
p('Adam lưu trung bình động của gradient và bình phương gradient, đồng thời hiệu chỉnh lệch ban đầu [10]. Khởi tạo trọng số theo phương sai 2/fan-in phù hợp với các tầng ReLU [11]. Cài đặt NumPy thực hiện trực tiếp các biểu thức; các framework cung cấp optimizer nhưng vẫn dùng cùng giá trị siêu tham số.')
p('Kiểm tra gradient bằng sai phân hữu hạn chỉ dùng tensor nhỏ và tránh vùng không trơn không cần thiết. Chênh lệch giữa đạo hàm giải tích và đạo hàm số được đối chiếu theo dung sai float32; bước kiểm tra này độc lập với việc mô hình đạt accuracy cao hay thấp.')

page('2.4. BatchNorm, residual và dropout')
sub('Batch Normalization')
p('BatchNorm chuẩn hóa theo thống kê minibatch rồi học hệ số scale γ và shift β [7]. Với ảnh, thống kê lấy theo N,H,W riêng từng kênh; với Dense, lấy theo batch riêng từng đặc trưng. Trong đánh giá, dùng running statistics đã tích lũy, không tính thống kê từ tập test.')
math_eq([r'\hat{x}=(x-\mu)/\sqrt{\sigma^2+\epsilon},\qquad y=\gamma\hat{x}+\beta',
         r'dx=\frac{\gamma}{\sqrt{\sigma^2+\epsilon}}\,[g-\mathrm{mean}(g)-\hat{x}\,\mathrm{mean}(g\hat{x})]'])
p('Ba triển khai dùng phương sai tổng thể. PyTorch mặc định có khác biệt trong cách cập nhật running variance; bài này viết lớp chuẩn hóa bằng tensor/autograd để khớp NumPy và Keras fused=False. Đây là lựa chọn phục vụ đối chiếu số học, cần ghi rõ khi so với nn.BatchNorm tiêu chuẩn.')
sub('Residual connection')
eq('y = ReLU(F(x) + x);  F(x) = BN(Conv3×3(x))')
p('Nhánh identity giúp gradient truyền qua phép cộng, đồng thời nhánh F học phần biến đổi cần bổ sung [8]. Block dùng cùng 16 kênh và cùng kích thước không gian nên không cần projection. Backward phải cộng gradient từ nhánh identity với gradient đi qua Conv và BN.')
sub('Dropout')
p('Trong train, dropout che ngẫu nhiên 25% đặc trưng ở phần phân loại; các phần tử giữ lại được chia cho 0,75 để bảo toàn kỳ vọng. Trong eval, dropout trở thành ánh xạ identity. Regularization có thể giảm phụ thuộc quá mức vào một số đặc trưng [9], nhưng cũng có thể làm học chậm khi mạng nhỏ hoặc số epoch ít.')
p('Phiên bản cuối đặt BN sau Dense 64 trước ReLU, ngoài các BN ở convolution. Thử nghiệm ban đầu thiếu lớp chuẩn hóa này cho train/validation loss cao ở CIFAR-100. Việc bổ sung được kiểm tra bằng gradient và áp dụng thống nhất ở cả ba backend; phần kết quả giữ lại pilot để đối chiếu.')

page('2.5. Kiến trúc cụ thể dùng trong bài')
table(['Khối','Baseline','Improved cuối'],[['Trích xuất 1','Conv 3×3, 8 → ReLU → Pool','Conv 3×3, 8 → BN → ReLU → Pool'],['Trích xuất 2','Conv 3×3, 16 → ReLU','Conv 3×3, 16 → BN → ReLU'],['Residual','Không có','ReLU(BN(Conv 16→16) + x)'],['Giảm kích thước','Pool → Flatten','Pool → Flatten'],['Phân loại','Dense 64 → ReLU → Dense C','Dense 64 → BN → ReLU → Dropout 0,25 → Dense C']],[90,180,213],caption='Hai kiến trúc được triển khai tương đương trong ba backend.')
table(['Dataset','Sau Pool 1','Sau Pool 2','Flatten'],[['MNIST','8 × 14 × 14','16 × 7 × 7','784'],['CIFAR-10/100','8 × 16 × 16','16 × 8 × 8','1.024']],[105,130,130,118],caption='Kích thước đặc trưng, không tính chiều batch.')
table(['Dataset','Baseline','Improved','Tăng tham số'],[[name.upper(),f"{get(name,'numpy','baseline')['parameter_count']:,}".replace(',','.'),f"{get(name,'numpy','improved')['parameter_count']:,}".replace(',','.'),'2.528'] for name in ['mnist','cifar10','cifar100']],[125,120,120,118],caption='Tham số trainable; dấu chấm trong số đếm phân tách hàng nghìn.')
p('Baseline có 2 convolution và 2 dense; improved có 3 convolution và 2 dense, thêm 4 BatchNorm. Số “lớp” có thể khác nếu tính cả activation và pooling, vì vậy báo cáo mô tả từng khối thay vì chỉ dùng một nhãn như CNN 3 lớp hay 5 lớp.')
p('Các cải tiến được đánh giá như một gói kiến trúc. Chênh lệch baseline/improved không đủ chứng minh đóng góp nhân quả riêng của residual, BN hoặc dropout. Pilot bổ sung một đối chiếu về BN ở Dense nhưng chưa phải ablation đầy đủ của mọi thành phần.')

page('CHƯƠNG 3. DỮ LIỆU VÀ TIỀN XỬ LÝ',chapter=True)
sub('3.1. Nguồn Kaggle và quy mô')
table(['Dataset','Train gốc','Test','Lớp / định dạng'],[['MNIST','60.000','10.000','10; ảnh xám 28×28'],['CIFAR-10','50.000','10.000','10; RGB 32×32'],['CIFAR-100','50.000','10.000','100 fine; RGB 32×32']],[112,105,90,176],caption='Quy mô gốc kiểm tra trên các file tải về.')
for idx,m in enumerate(manifests,4):p(f'[{idx}] '+m['dataset'].upper()+': '+link(m['kaggle_url'],m['kaggle_url'])+'.','small')
p('CIFAR-10 và CIFAR-100 là các bộ ảnh của Krizhevsky, Nair và Hinton [13]. MNIST dùng bản CSV có cột nhãn và 784 cột pixel. CIFAR dùng bản Python, chứa mảng ảnh, nhãn và metadata; chương trình chỉ cho phép các đối tượng NumPy cần thiết khi đọc định dạng pickle.')
sub('3.2. Quy trình nạp dữ liệu')
p('Script tải ZIP công khai từ Kaggle, kiểm tra đường dẫn giải nén không vượt thư mục đích, lưu SHA-256 và danh sách file. Mảng ảnh được chuyển về NCHW, nhãn int64, tên lớp dạng chuỗi. File NPZ chuẩn hóa cấu trúc giúp cả ba backend dùng cùng dữ liệu đầu vào.')
p('Tập validation được tách bằng shuffle theo lớp với seed 42. 10% mỗi lớp được làm tròn riêng; vì vậy MNIST có 5.999 mẫu validation. Không chuyển ảnh giữa train và test chính thức. Tất cả chỉ số split được xuất ra file để đối chiếu, thay vì chỉ ghi seed và chia lại tùy framework.')
table(['Dataset','Train','Validation','Test'],[[m['dataset'].upper(),f"{m['train']:,}",f"{m['validation']:,}",f"{m['test']:,}"] for m in manifests],[135,116,116,116],caption='Phân chia thực tế dùng trong 18 thí nghiệm.')

page('3.3. Kiểm tra chất lượng và nguy cơ rò rỉ')
p('Các ảnh có kiểu uint8 và pixel trong [0,255]. Nhãn nằm trong miền lớp hợp lệ. Train và validation không giao nhau theo chỉ số; tổng số chỉ số bằng toàn bộ tập training gốc. MNIST và CIFAR-10 không có ảnh pixel trùng hoàn toàn giữa các phần theo phép kiểm tra SHA-256 đã thực hiện.')
table(['Dataset','Ảnh train gốc duy nhất','Hash chung train-val','Hash chung train-test','Hash chung val-test'],[[a['dataset'].upper(),a['unique_training_images'],a['exact_image_hashes_shared_train_validation'],a['exact_image_hashes_shared_train_test'],a['exact_image_hashes_shared_validation_test']] for a in audits],[90,103,98,98,94],caption='Kiểm tra trùng hoàn toàn theo nội dung pixel; số liệu là số hash chung.')
p('CIFAR-100 có 49.986 ảnh pixel duy nhất trong 50.000 mẫu training gốc. Có một số nội dung ảnh xuất hiện ở hai phần khác nhau mặc dù chỉ số mẫu khác nhau. Điều này làm suy yếu giả thiết độc lập hoàn toàn của các phần dữ liệu; riêng các ảnh trùng train-validation có thể ảnh hưởng nhẹ việc chọn checkpoint.')
p('Bài giữ nguyên benchmark và công khai hạn chế, không thay đổi split hoặc lọc mẫu sau khi xem mô hình dự đoán đúng/sai. Phần phân tích độ nhạy tính thêm accuracy trên các ảnh test không trùng training gốc; đây là kiểm tra bổ sung, không thay thế chỉ số benchmark chính thức.')
p('Exact hash chỉ phát hiện ảnh có pixel giống hệt, không phát hiện các bản cắt, đổi màu hoặc ảnh gần giống. Do đó kết quả kiểm tra bằng 0 không đồng nghĩa không còn mọi dạng trùng lặp ngữ nghĩa. Nghiên cứu tiếp theo nên xây dựng split theo nhóm nội dung từ đầu và đánh giá trên một holdout mới.')
sub('Kiểm soát tiền xử lý')
p('Chia 255 là biến đổi cố định, không dùng thống kê test. Không chuẩn hóa theo mean/std của toàn bộ dữ liệu, không tăng cường dữ liệu trước khi chia tập. BatchNorm ở test dùng thống kê đã học, dropout tắt. Những lựa chọn này tránh việc bước chuẩn bị hoặc chế độ mô hình vô tình dùng thông tin test để huấn luyện.')

for number,name in enumerate(['mnist','cifar10','cifar100'],4):
    page(f'3.{number}. Khảo sát {name.upper()}')
    if name=='mnist':
        p('MNIST gồm chữ số viết tay từ 0 đến 9 trên ảnh xám 28×28. Nền đơn giản và đối tượng được căn tương đối đồng nhất giúp mạng nhỏ học nhanh. Tuy vậy, kiểu chữ khác nhau, nét mờ hoặc chữ số có hình dạng gần nhau vẫn gây lỗi.')
    elif name=='cifar10':
        p('CIFAR-10 có 10 lớp: airplane, automobile, bird, cat, deer, dog, frog, horse, ship và truck. Ảnh RGB kích thước nhỏ chứa thay đổi về nền, tư thế, ánh sáng và tỷ lệ đối tượng. So với MNIST, mạng phải phân biệt cấu trúc tự nhiên thay vì chủ yếu hình dạng nét trên nền đơn giản.')
    else:
        p('CIFAR-100 tăng bài toán lên 100 lớp fine với 500 ảnh training gốc và 100 ảnh test mỗi lớp. Sau chia validation, mỗi lớp còn 450 ảnh để cập nhật trọng số. Nhiều lớp thuộc cùng nhóm ngữ nghĩa hoặc có màu/kết cấu gần nhau; độ phân giải 32×32 làm mất các dấu hiệu phân biệt nhỏ.')
    fig(f'{name}_samples.png','Hai ảnh mỗi lớp minh họa lấy từ phần train.'+(' Với CIFAR-100, chọn 10 lớp có chỉ số cách nhau 10, không đại diện toàn bộ 100 lớp.' if name=='cifar100' else ''))
    fig(f'{name}_distribution.png','Phân bố số mẫu trong phần train sau chia validation; chỉ số lớp giữ theo metadata dữ liệu.')
    if name=='mnist':p('Số mẫu mỗi chữ số khác nhau nhưng không quá chênh lệch. Lớp phổ biến nhất trong phần train là chữ số 1; luôn dự đoán 1 chỉ đạt 11,35% test accuracy. Macro-F1 giúp cho các lớp trọng số ngang nhau khi tổng hợp, bổ sung cho accuracy chịu ảnh hưởng số mẫu. Toàn bộ ảnh giữ nguyên 28×28, không cần phóng to để khớp CIFAR.')
    elif name=='cifar10':p('Sau chia, mỗi lớp có 4.500 ảnh train, 500 validation và 1.000 test. Vì test cân bằng, macro-recall bằng accuracy; macro-F1 vẫn cung cấp thông tin khác do chịu ảnh hưởng precision của từng lớp. Không cần class weighting để bù mất cân bằng ở benchmark này. Dữ liệu có cấu trúc không gian rõ ràng, phù hợp Conv2D hơn ba bộ dạng bảng/văn bản ban đầu.')
    else:p('Mỗi lớp có 450 ảnh train, 50 validation và 100 test. Số mẫu trên mỗi lớp chỉ bằng một phần mười CIFAR-10 dù số ảnh toàn bộ bằng nhau. Vì vậy cần kiểm tra cả macro-F1, top-5 và các cặp lớp hay nhầm. Các số lượng và hình minh họa ở đây được tạo trực tiếp từ file tải về; hạn chế trùng pixel được giữ trong phần kiểm toán dữ liệu.')

page('CHƯƠNG 4. CÀI ĐẶT VÀ KIỂM CHỨNG',chapter=True)
sub('4.1. CNN từ đầu bằng NumPy')
p('Module numpy_cnn.py chỉ dùng NumPy cho phép tính mạng và cập nhật. Mỗi lớp cung cấp forward, backward, danh sách tham số/gradient và state. CNN gọi các lớp theo thứ tự ở forward và đảo thứ tự ở backward. Thiết kế này thể hiện trực tiếp quy tắc hợp thành hàm và giúp kiểm tra từng lớp độc lập.')
table(['Thành phần','Cách triển khai'],[['Conv2D','sliding_window_view → im2col → nhân ma trận; backward tính dW, db và col2im'],['ReLU / MaxPool','Lưu mask hoặc argmax ở train; gradient chỉ về vị trí đã chọn'],['BatchNorm 2D/1D','Tự tính mean/variance và đạo hàm; cùng state running cho eval'],['Residual','Lưu nhánh Conv+BN; cộng gradient nhánh identity'],['Dense / Dropout','Nhân ma trận; inverted dropout dùng RNG'],['Loss / optimizer','Stable softmax cross-entropy và Adam có bias correction']],[115,368],caption='Các khối thuật toán từ đầu.')
p('Lớp convolution dùng nhân ma trận của NumPy để tăng tốc, nhưng công thức forward/backward do chương trình tự tổ chức. Việc sử dụng BLAS phía dưới NumPy không tương đương dùng autograd: chương trình vẫn phải tạo dW, db, dX và cập nhật đúng cho từng tham số.')
p('Ở train, lớp lưu các giá trị cần cho backward; ở eval, không tạo cache không cần thiết. Checkpoint NPZ lưu trọng số và running statistics. Sau khi huấn luyện, chương trình khôi phục checkpoint tốt nhất theo validation rồi mới tính test logits.')
p('Bài toán thử nhỏ và kiểm tra sai phân hữu hạn giúp phát hiện lỗi cài đặt trước các lượt huấn luyện đầy đủ. Đạt loss giảm trên dữ liệu thật là bằng chứng mô hình học được, nhưng không thay thế kiểm chứng đạo hàm vì một số lỗi gradient vẫn có thể cho đường loss giảm.')

page('4.2. Cài đặt bằng PyTorch')
p('TorchCNN kế thừa nn.Module, biểu diễn các khối bằng nn.Sequential, nn.Conv2d, nn.Linear, activation, pooling và dropout. Residual được viết thành module riêng. PyTorch tự theo dõi đồ thị phép toán và tính gradient khi gọi loss.backward().')
eq('zero_grad → forward → cross_entropy → backward → optimizer.step')
p('Trước mỗi minibatch, xóa gradient cũ bằng zero_grad(set_to_none=True) để tránh cộng gradient ngoài ý muốn. Trong đánh giá, gọi eval và no_grad để tắt dropout, dùng running statistics và không giữ đồ thị gradient. Dữ liệu batch được chuyển từ NumPy sang device CUDA.')
p('Trọng số convolution của PyTorch và NumPy đều dùng thứ tự Cout,Cin,K,K. Dense của NumPy có Din,Dout, trong khi nn.Linear lưu Dout,Din nên phải chuyển vị. Nếu quên chuyển vị hoặc đổi thứ tự Flatten, chương trình có thể vẫn chạy nhưng không còn biểu diễn cùng mạng.')
p('PopulationBatchNorm viết bằng phép toán tensor và autograd để thống nhất cách cập nhật phương sai chạy. Đây không phải yêu cầu thông thường của mọi mô hình PyTorch; nó phục vụ thí nghiệm cần đối chiếu số học giữa ba triển khai. Các kiểm tra train/eval xác nhận hành vi của lớp này so với NumPy.')
table(['Nhiệm vụ','Cơ chế'],[['Chế độ train','model.train(); tính gradient và cập nhật Adam'],['Chế độ test','model.eval(); torch.no_grad()'],['Checkpoint','state_dict lưu bằng torch.save; tải weights_only=True'],['Thiết lập GPU','Tắt TF32; cuDNN deterministic; không benchmark tự chọn thuật toán'],['Giới hạn tái lập','Một seed không bảo đảm mọi máy/kernel cho cùng từng bit']],[125,358],caption='Các điểm cài đặt PyTorch liên quan đến tính đúng đắn.')

page('4.3. Cài đặt bằng TensorFlow/Keras')
p('Mô hình TensorFlow dùng Functional API. Input có thứ tự NHWC theo thói quen Keras. Convolution và pooling giữ cùng hình học với hai triển khai còn lại. Trước Flatten, Permute chuyển NHWC về NCHW để Dense nhìn cùng thứ tự đặc trưng.')
p('Vòng lặp tùy chỉnh dùng GradientTape để lấy đạo hàm của sparse softmax cross-entropy theo trainable_weights. tf.function biên dịch bước train và infer nhằm giảm chi phí gọi Python. Vì epoch đầu có thêm thời gian biên dịch/khởi tạo kernel, không nên coi thời gian của nó như các epoch sau.')
eq('GradientTape: logits → loss → gradient(loss, weights) → apply_gradients')
p('Keras BatchNormalization dùng momentum 0,9 và fused=False, epsilon 1e-5. Giá trị momentum ở Keras là trọng số của thống kê cũ, trong khi cách diễn đạt thường dùng ở NumPy/PyTorch có thể là trọng số của batch mới. Bài ghi rõ công thức running = 0,9 × cũ + 0,1 × batch để tránh hiểu nhầm.')
p('Checkpoint H5 lưu weights và running statistics; sau chọn epoch, load_weights khôi phục mô hình để đánh giá. Optimizer Adam dùng learning rate 0,001 và epsilon 1e-8, nhưng vị trí epsilon trong công thức hiệu chỉnh có thể khác bản NumPy/PyTorch. Cộng với sai số GPU và dropout, điều này giải thích vì sao cùng khởi tạo chưa đủ tạo đường học giống hệt.')
table(['Vấn đề','Cách xử lý'],[['Khác layout','NCHW → NHWC khi nhập; Permute trước Flatten'],['GPU Windows','TensorFlow 2.10.1, CUDA 11.2, cuDNN 8.1'],['Bộ nhớ GPU','Memory growth được cấu hình trước tạo model'],['Tách framework','Huấn luyện trong subprocess riêng'],['Trọng số ban đầu','Conv chuyển OIHW → HWIO; Dense giữ Din,Dout']],[125,358],caption='Các chi tiết bảo đảm TensorFlow tương ứng kiến trúc gốc.')

page('4.4. Kiểm tra gradient và đối chiếu giữa backend')
p('Bộ kiểm tra NumPy gồm 9 test: gradient convolution, dense, BatchNorm ảnh, BatchNorm sau Dense, max pooling, residual; ổn định loss; shape/số tham số; khả năng học một bài toán nhỏ. Tất cả đã chạy thành công. Sai phân hữu hạn kiểm tra các phần tử được chọn với bước 0,002 và dung sai phù hợp float32.')
verifications=read('results/verification_pytorch.json')+read('results/verification_tensorflow.json')
table(['Backend so với NumPy','Số cấu hình','Sai số logits lớn nhất','Sai số gradient Conv1 lớn nhất'],[[b,6,f"{max(v['max_absolute_errors']['train_logits'] for v in verifications if v['backend']==b):.2e}",f"{max(v['max_absolute_errors']['conv1_gradient'] for v in verifications if v['backend']==b):.2e}"] for b in ['pytorch','tensorflow']],[145,83,125,130],caption='Đối chiếu trên CPU với dropout tắt; cả 12 cấu hình đều đạt dung sai.')
p('Mỗi cấu hình so sánh logits khi train, logits khi eval sau cập nhật running statistics, gradient đầu vào và gradient trọng số Conv1. Kiểm tra dùng batch nhỏ chung, trọng số ban đầu chung, và tắt dropout để bỏ khác biệt ngẫu nhiên. Gradient Conv1 ở cuối chuỗi lan truyền đi qua phần lớn cấu trúc mạng, nên hữu ích hơn chỉ so shape.')
p('Không kiểm tra bitwise equality. Dung sai assert_allclose là atol 2e-4 và rtol 3e-4; sai số quan sát nhỏ hơn nhiều so với giới hạn này. Đối chiếu này xác nhận phép tính cài đặt tương ứng trong các trường hợp thử, không khẳng định mọi hành vi ở mọi đầu vào đã được chứng minh tuyệt đối.')
sub('Kiểm chứng đầu ra sau huấn luyện')
p('Script phân tích đọc lại logits và nhãn của cả 18 lượt, tính lại metric, đối chiếu confusion matrix CSV và kiểm tra best_epoch đúng vị trí validation loss nhỏ nhất. Mỗi history có đủ số epoch đã cấu hình. Các notebook được thực thi và kiểm tra không có ô lỗi trước khi đóng gói.')
p('File verification JSON, history CSV và checkpoint đi kèm repository giúp người đọc truy vết các khẳng định thay vì chỉ xem bảng đã làm tròn trong PDF.')

for chapter_index,name in enumerate(['mnist','cifar10','cifar100'],1):
    page(('CHƯƠNG 5. KẾT QUẢ THỰC NGHIỆM' if chapter_index==1 else f'5.{2*chapter_index-1}. Kết quả {name.upper()}'),chapter=chapter_index==1)
    if chapter_index==1:sub('5.1. Kết quả MNIST')
    data=subset(name)
    table(['Backend','Mô hình','Acc. (%)','Macro-F1','Top-5 (%)','Epoch tốt'],[[r['backend'],r['variant'],pct(r['accuracy']),dec(r['macro_f1']),pct(r['top5_accuracy']),r['best_epoch']] for r in data],[90,78,77,85,83,70],caption=f'Kết quả test {name.upper()} trên 10.000 ảnh; epoch chọn bằng validation loss.')
    fig(f'{name}_curves.png',f'Loss và accuracy train/validation của sáu cấu hình {name.upper()}. Nét đứt: baseline; nét liền: improved; xám: train; màu: validation.')
    gains=[100*(get(name,b,'improved')['accuracy']-get(name,b,'baseline')['accuracy']) for b in ['numpy','pytorch','tensorflow']]
    p('Mức thay đổi accuracy khi dùng improved lần lượt ở NumPy, PyTorch, TensorFlow là '+', '.join(f'{g:+.2f}'.replace('.',',') for g in gains)+' điểm phần trăm. Đây là các lượt chạy riêng seed 42; không phải trung bình nhiều seed. Chọn theo validation loss có thể lấy epoch khác epoch cuối hoặc epoch có validation accuracy cao nhất.','small')
    page(f'5.{2*chapter_index}. Phân tích lỗi {name.upper()}')
    details=read(f'results/{name}_error_analysis.json')
    p(f"Mô hình minh họa là {details['representative_backend']} {details['representative_variant']}, được chọn vì có best validation loss thấp nhất trong sáu lượt của dataset. Các ảnh bên dưới là dự đoán sai tự tin nhất; không dùng chúng để thay đổi mô hình sau khi hoàn tất thí nghiệm chính.")
    fig(f'{name}_errors.png','Mười ảnh dự đoán sai có confidence cao nhất. T là nhãn thật, P là nhãn dự đoán; confidence là softmax lớn nhất, chưa được hiệu chỉnh xác suất.')
    table(['Nhãn thật','Bị nhầm thành','Số ảnh'],[[d['true'],d['predicted'],d['count']] for d in details['worst_pairs'][:5]],[190,190,103],caption='Năm hướng nhầm lẫn nhiều nhất trong mô hình minh họa.')
    if name=='mnist':p('Các lỗi còn lại thường xuất hiện ở hình dạng nét khó phân biệt hoặc cách viết khác thông thường. Confidence cao ở một dự đoán sai cho thấy softmax không đồng nghĩa độ chắc chắn đã hiệu chỉnh. Chỉ từ các ảnh này chưa thể kết luận nhãn dataset sai; cần xem lại dữ liệu gốc và tiêu chí gán nhãn trước khi chỉnh nhãn.')
    elif name=='cifar10':p('Các nhóm động vật hoặc phương tiện có thể chia sẻ hình dạng, màu và nền. CNN nhỏ nhìn ảnh 32×32 nên chi tiết phân biệt bị hạn chế. Cần đối chiếu tần suất nhầm với toàn bộ số mẫu mỗi lớp; một số ví dụ nổi bật không đủ chứng minh mô hình luôn dựa vào nền ảnh.')
    else:p('Bài toán 100 lớp đòi hỏi phân biệt tinh hơn trong cùng độ phân giải và cùng phần trích xuất nhỏ. Macro-F1 thấp hơn accuracy ở một số lượt cho thấy chất lượng không đồng đều giữa lớp và phân phối dự đoán. Top-5 bổ sung góc nhìn về việc nhãn thật còn nằm trong nhóm ứng viên mạnh hay đã bị loại xa.')

page('5.7. Confusion matrix và recall từng lớp')
fig('cifar10_confusion.png','Confusion matrix CIFAR-10 chuẩn hóa theo nhãn thật. Ô đường chéo là recall của từng lớp; màu đậm ngoài đường chéo biểu thị nhầm lẫn có hệ thống.',width=375)
p('Ma trận lưu đầy đủ dưới dạng CSV cho từng lượt, bao gồm CIFAR-100 kích thước 100×100. Hàng là nhãn thật, cột là nhãn dự đoán. Khi chuẩn hóa theo hàng, tổng mỗi hàng bằng 1; nhờ vậy có thể đọc tỷ lệ nhầm thay vì chỉ số lượng.')
p('Notebook so sánh kèm hình ma trận MNIST/CIFAR-100 và biểu đồ recall của các lớp thấp nhất. Không thu nhỏ toàn bộ 100 nhãn vào một bảng chữ khó đọc trong báo cáo; dữ liệu chi tiết vẫn có thể xem trong file đầu ra và hình ở repository.')

page('5.8. So sánh tổng hợp và chi phí')
fig('accuracy_comparison.png','Accuracy test của 18 cấu hình. Cùng một thang phần trăm giúp thấy chênh lệch độ khó giữa ba dataset.')
table(['Dataset','Backend','Δ accuracy (đpt)','Train baseline (s)','Train improved (s)'],[[name.upper(),b,f"{100*(get(name,b,'improved')['accuracy']-get(name,b,'baseline')['accuracy']):+.2f}",f"{get(name,b,'baseline')['train_seconds']:.1f}",f"{get(name,b,'improved')['train_seconds']:.1f}"] for name in ['mnist','cifar10','cifar100'] for b in ['numpy','pytorch','tensorflow']],[88,90,100,100,105],caption='Chênh lệch improved-baseline và tổng thời gian train, chưa gồm validation/test.')
p('Improved thêm 2.528 tham số và nhiều phép tính hơn. Lợi ích cần được cân nhắc cùng độ phức tạp và ngân sách huấn luyện. Thời gian CPU/GPU không đo trong điều kiện cô lập, nên bảng không dùng để xếp hạng tốc độ backend. Khác biệt nhỏ về accuracy cũng chưa cho thấy ưu thế ổn định khi chưa lặp nhiều seed.','small')

page('5.9. Thử nghiệm ban đầu và điều chỉnh phần Dense')
fig('pilot_dense_bn.png','Validation loss của improved ban đầu và phiên bản cuối trong TensorFlow. Phiên bản cuối thêm BatchNorm sau Dense 64; các điều kiện huấn luyện còn lại giữ cùng cấu hình.')
pilot=list(csv.DictReader((ROOT/'results/pilot_comparison.csv').open(encoding='utf-8')))
table(['Dataset','Kiến trúc','Train loss cuối','Val loss tốt nhất'],[[r['dataset'].upper(),'Thiếu BN Dense' if 'no Dense' in r['architecture'] else 'Có BN Dense',dec(r['final_train_loss']),dec(r['best_val_loss'])] for r in pilot],[96,147,120,120],caption='Đối chiếu train/validation của thử nghiệm kiến trúc ban đầu.')
p('Ở CIFAR-100, bản thử có cả train và validation loss cao, phù hợp dấu hiệu underfitting trong ngân sách đã đặt. Bổ sung BN sau Dense cải thiện khả năng tối ưu của phần phân loại. Đây là quan sát thực nghiệm của cấu hình cụ thể; chưa chứng minh mọi mạng CNN đều cần BN ở Dense hoặc BN là nguyên nhân duy nhất của mọi chênh lệch.')
p('Bốn lượt pilot và source cũ được lưu riêng. Một số test metrics của pilot đã được tạo trước khi điều chỉnh; vì vậy toàn bộ quy trình mang tính khám phá, không được trình bày như một thử nghiệm holdout hoàn toàn mù. Quyết định điều chỉnh dựa trên train/validation; bảng chính chỉ gồm kiến trúc cuối đã thống nhất cho cả ba backend.')

page('5.10. Giới hạn và độ tin cậy của kết quả')
sub('Một seed, ngân sách nhỏ và kiến trúc gộp')
p('Mỗi cấu hình chính chỉ chạy một seed, chưa có khoảng tin cậy giữa các lần huấn luyện. Mạng dùng 8/16 kênh, Dense 64 và 5-12 epoch để phù hợp phần NumPy từ đầu. Accuracy trên CIFAR không đại diện cho khả năng tốt nhất của CNN hiện đại. Chưa có augmentation, lịch learning rate hoặc ablation tách riêng từng thành phần.')
sub('Ảnh trùng hoàn toàn trong CIFAR-100')
sens=read('results/cifar100_duplicate_sensitivity.json')
table(['Backend','Mô hình','Test gốc (%)','Bỏ ảnh trùng (%)'],[[r['backend'],r['variant'],pct(r['official_accuracy']),pct(r['unseen_pixel_accuracy'])] for r in sens],[110,110,132,131],caption=f"Độ nhạy với {sens[0]['excluded_test_images']} ảnh test có pixel đã xuất hiện trong training gốc; còn {sens[0]['retained_test_images']} ảnh.")
p('Đây là phép kiểm tra bổ sung theo nội dung pixel, không thay thế benchmark. Các trùng lặp train-validation vẫn có thể tác động đến chọn epoch; near-duplicate chưa được kiểm tra. Nghiên cứu nghiêm ngặt hơn cần chia nhóm ảnh trước mọi lượt thử và dùng một tập holdout mới.')
sub('Diễn giải learning curves và confidence')
p('Train metrics được tích lũy trong khi trọng số đang đổi và dropout đang bật; validation đo sau epoch ở eval. Vì vậy validation accuracy cao hơn train accuracy không tự động là lỗi. Confidence softmax chưa được calibration, nên không dùng trực tiếp làm xác suất tin cậy cho quyết định thực tế.')

page('CHƯƠNG 6. KẾT LUẬN VÀ TÁI LẬP',chapter=True)
sub('6.1. Kết luận từ bài thực hành')
p('Bài làm hoàn thành chuỗi từ lý thuyết hợp thành hàm đến forward/backward NumPy và hai framework. Các kiểm tra gradient, đối chiếu số học và kết quả học trên dữ liệu thật cung cấp bằng chứng bổ trợ cho tính đúng đắn của triển khai. Cùng một CNN có thể biểu diễn tương ứng qua ba hệ công cụ khi kiểm soát layout, thứ tự trọng số và trạng thái train/eval.')
for name in ['mnist','cifar10','cifar100']:
    vals=[r['accuracy'] for r in subset(name)]
    p(f"Trên {name.upper()}, sáu cấu hình đạt accuracy trong khoảng {pct(min(vals))}% đến {pct(max(vals))}%. Khoảng này mô tả các lượt chạy trong bài, không phải độ biến thiên thống kê của một mô hình. Độ khó tăng rõ khi chuyển từ chữ số nền đơn giản sang ảnh tự nhiên và 100 nhãn fine.")
p('Cải tiến có ích khi kết hợp với kiểm tra đường học và điều chỉnh hợp lý. Thử nghiệm ban đầu trên CIFAR-100 cho thấy thêm residual và dropout chưa đủ; chuẩn hóa phần Dense giúp bản cuối tối ưu tốt hơn trong cùng số epoch. Bài giữ cả kết quả thử chưa tốt để tránh trình bày một quá trình chỉ có thành công.')
sub('6.2. Hướng phát triển')
p('Có thể mở rộng theo thứ tự: xây dựng split theo nhóm nội dung, lặp nhiều seed, thử augmentation chỉ trên train, dùng lịch learning rate và thực hiện ablation BN/residual/dropout. Một kiến trúc global average pooling có thể giảm phần Dense; số kênh lớn hơn có thể cải thiện biểu diễn nhưng làm NumPy tốn thời gian hơn.')
p('Nếu giảng viên yêu cầu thêm ba dataset trong slide, cần làm một nhánh CNN 1D riêng: diabetes và nhà ở cần đánh giá giả định lân cận của cột; văn bản đánh giá quần áo cần tokenize và embedding. Không áp dụng reshape thành ảnh tùy tiện chỉ để dùng Conv2D.')

page('6.3. Hướng dẫn chạy bằng Anaconda/Jupyter')
p('README trong repository là hướng dẫn đầy đủ và environment.yml ghim phiên bản thư viện. Trên Windows, tạo môi trường assignment04 từ file này, kích hoạt môi trường, cài gói PyTorch cu121 nếu dùng NVIDIA GPU, rồi đăng ký kernel Python (Assignment 04). Đặt CNN_CUDA_DIR vào Library/bin của môi trường CUDA phù hợp.')
table(['Bước','Thực hiện'],[['1. Lấy source','Clone repository hoặc giải nén gói bài nộp'],['2. Chuẩn bị môi trường','conda env create -f environment.yml; conda activate assignment04'],['3. Dữ liệu','Đặt CNN_DATA_DIR; chạy python -m src.download_data'],['4. Mở notebook','jupyter lab; chọn kernel Python (Assignment 04)'],['5. Đọc theo thứ tự','00 Discover → 01 Datasets → 02 NumPy → 03 PyTorch → 04 TensorFlow → 05 Compare'],['6. Huấn luyện lại','RETRAIN=True trong notebook hoặc python run_all.py --force'],['7. Tạo bảng/đồ thị','python -m src.analyze']],[110,373],caption='Quy trình chạy chính; các lệnh chi tiết có trong README.')
p('Notebook mặc định đọc kết quả đã huấn luyện để người chấm xem nhanh. Ô chạy ghi rõ khi tái sử dụng kết quả; đầu ra không giả làm log huấn luyện mới. Đặt RETRAIN=True sẽ chạy thật các cấu hình và ghi đè thư mục kết quả tương ứng. Nếu cần lưu nhiều seed, nên sao lưu kết quả hoặc đổi cơ chế đặt tên trước khi chạy.')
p('run_all.py chạy từng backend trong tiến trình riêng. Một cấu hình đã có metrics.json được bỏ qua, trừ khi dùng --force. Lượt bị ngắt chưa hoàn tất metric sẽ được chạy lại từ đầu; checkpoint hiện lưu trạng thái suy luận, chưa lưu đầy đủ optimizer để tiếp tục một epoch đang dở.')
p('Nếu API Kaggle yêu cầu xác thực hoặc thay đổi, tải ba ZIP qua các link đã nêu, đặt đúng tên trong CNN_DATA_DIR rồi chạy lại bước chuẩn bị. Không cần tải dữ liệu gốc từ GitHub; repository lưu link, checksum và split, còn dữ liệu lớn được quản lý ở thư mục riêng.')

page('6.4. Cấu trúc bài nộp và khả năng truy vết')
table(['Thư mục / file','Nội dung'],[['notebooks/','6 notebook thuật toán/phân tích có đầu ra thực thi'],['src/','CNN NumPy, PyTorch, TensorFlow; tải dữ liệu, train và phân tích'],['tests/','Sai phân hữu hạn và đối chiếu giữa framework'],['results/','18 thư mục thí nghiệm; split, audit, summary và verification'],['figures/','Ảnh minh họa, learning curves, confusion matrix, phân tích lỗi'],['experiments/pilot_v1/','Mã cũ và 4 lượt thử kiến trúc ban đầu'],['report/','PDF hoàn chỉnh, bìa gốc và script tạo báo cáo'],['README.md / environment.yml','Nguồn dữ liệu, môi trường, hướng dẫn chạy']],[160,323],caption='Các thành phần đi kèm PDF.')
sub('Nội dung một thư mục thí nghiệm')
p('config.json ghi dữ liệu, backend, kiến trúc, siêu tham số, phiên bản và thiết bị. history.csv có một dòng mỗi epoch. metrics.json tổng hợp test; predictions.csv giữ test_id, nhãn thật, nhãn dự đoán và confidence. test_outputs.npz giữ logits/labels để tính lại metric. confusion_matrix.csv giữ số đếm theo cặp lớp. weights lưu checkpoint có validation loss thấp nhất.')
p('Bảng PDF làm tròn để dễ đọc; file CSV/JSON giữ độ chính xác gốc. Việc tính lại metric từ logits cho phép phát hiện chép nhầm kết quả hoặc nhầm checkpoint. Các source ZIP và mảng đã chuẩn bị có SHA-256 trong manifest; dữ liệu gốc và tài liệu sách/slide không được đưa vào repository.')
p('Liên kết source: '+link(gh,gh)+'. Bìa sử dụng bản PDF xuất từ mẫu Google Docs do người nộp cung cấp. Nội dung chuyên môn, số liệu và hình thực nghiệm được tạo cho Assignment 04.')

page('TÀI LIỆU THAM KHẢO',chapter=True)
refs=[
('[1]','Slide học phần Phát triển các hệ thống thông minh, intel_sys_dev_slide_04_updated_11.9.pdf, tài liệu do giảng viên cung cấp.',''),
('[2]','Deep Learning CNN Function Composition Tutorial, tài liệu hướng dẫn được cung cấp, đặc biệt phần CNN và gợi ý thực hành.',''),
('[3]','François Chollet. Deep Learning with Python, 2nd edition. Manning, 2021. Tham khảo các phần đánh giá/workflow (trang in 133-137, 160-165), CNN (202-205, 221-225) và kiến trúc hiện đại (251-262).',''),
('[4]','MNIST in CSV, Kaggle, oddrationale.','https://www.kaggle.com/datasets/oddrationale/mnist-in-csv'),
('[5]','CIFAR-10 Python, Kaggle, pankrzysiu.','https://www.kaggle.com/datasets/pankrzysiu/cifar10-python'),
('[6]','CIFAR-100, Kaggle, fedesoriano.','https://www.kaggle.com/datasets/fedesoriano/cifar100'),
('[7]','S. Ioffe và C. Szegedy. Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift, 2015.','https://arxiv.org/abs/1502.03167'),
('[8]','K. He, X. Zhang, S. Ren và J. Sun. Deep Residual Learning for Image Recognition, 2015.','https://arxiv.org/abs/1512.03385'),
('[9]','N. Srivastava và cộng sự. Dropout: A Simple Way to Prevent Neural Networks from Overfitting. JMLR 15, 2014.','https://www.jmlr.org/papers/v15/srivastava14a.html'),
('[10]','D. P. Kingma và J. Ba. Adam: A Method for Stochastic Optimization, 2014.','https://arxiv.org/abs/1412.6980'),
('[11]','K. He và cộng sự. Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification, 2015.','https://arxiv.org/abs/1502.01852'),
('[12]','TensorFlow. Install TensorFlow with pip; ghi chú GPU Windows bản địa.','https://www.tensorflow.org/install/pip'),
('[13]','A. Krizhevsky, V. Nair và G. Hinton. CIFAR-10 and CIFAR-100 datasets.','https://www.cs.toronto.edu/~kriz/cifar.html')]
for number,description,url in refs:
    p(number+' '+description+(' '+link(url,'Nguồn trực tuyến') if url else ''),'small')
p('Nguồn trực tuyến được kiểm tra ngày 13/09/2026. Báo cáo mẫu A3_05_dungvt.194.pdf chỉ được tham khảo về bố cục, không dùng làm nguồn số liệu thực nghiệm.','small')

OUT.mkdir(exist_ok=True);body=OUT/'_body.pdf';Report(body).multiBuild(S)
writer=PdfWriter();writer.append(str(OUT/'cover.pdf'));writer.append(str(body))
writer.add_metadata({'/Title':'Assignment 04 - CNN - Nguyễn Ngọc Hoàng Nam - B23DCCN585','/Author':'Nguyễn Ngọc Hoàng Nam','/Subject':'MNIST, CIFAR-10, CIFAR-100; NumPy, PyTorch, TensorFlow'})
target=OUT/'Assignment04_NguyenNgocHoangNam_B23DCCN585.pdf'
with target.open('wb') as f:writer.write(f)
body.unlink()
print(target);print('Pages:',len(PdfReader(target).pages))
