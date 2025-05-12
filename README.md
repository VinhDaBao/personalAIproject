# 23110366 - Trịnh Quốc Công Vinh
## 1. Mục tiêu
### 1.1. Nhóm thuật toán

Gồm 6 nhóm thuật toán :
  - Tìm kiếm không có thông tin: BFS, DFS, IDS, UCS.
  - Tìm kiếm có thông tin: Greedy Best-First Search, A*, IDA*
  - Tìm kiếm cục bộ: Simple Hill Climbing, Steepest Ascent Hill Climbing, Stochastic Hill Climbing, Beam, Genetic Algorithm
  - Tìm kiếm trong môi trường phức tạp: And-Or Search, Belief, Partially
  - Tìm kiếm trong môi trường ràng buộc: Kiểm thử, Backtracking, AC-3
  - Học tăng cường: Q-Learning

Các nhóm thuật toán dùng để giải bài toán 8 - Puzzle, mỗi nhóm đều có điểm mạnh, điểm yếu và điểm chung giữa các thuật toán chung 1 nhóm.
### 1.2. Mục tiêu bài tập

Hiểu bản chất của từng nhóm thuật toán, điểm chung, điểm mạnh, điểm yếu của từng nhóm.
So sánh hiệu năng và tính phù hợp

Hiểu sự khác biệt về:
  - Hiệu quả thời gian và bộ nhớ
  - Độ chính xác (tìm lời giải tối ưu hay không)
  - Tính khả thi với bài toán có không gian trạng thái lớn

Ứng dụng vào bài toán cụ thể: 8-puzzle.
Áp dụng từng thuật toán để giải cùng một bài toán → so sánh kết quả.

Củng cố kỹ năng lập trình và tư duy giải quyết vấn đề. Hiểu và hiện thực các thuật toán.
  - Thực hành cài đặt cấu trúc dữ liệu, hàng đợi, cây tìm kiếm...
  - Đánh giá khả năng mở rộng và cải tiến thuật toán.
## 2. Nội dung
### 2.1. Các thuật toán tìm kiếm không có thông tin
Các thành phần chính bao gồm mảng hai chiều: trạng thái bắt đầu và kết thúc, và solution.
Cấu trúc lưu trữ các trạng thái đã duyệt qua như queue, stack.

BFS:

![BFS](https://github.com/user-attachments/assets/81a9e1ed-8dc2-4ed4-87a8-b9107b4651f9)

DFS:

![DFS](https://github.com/user-attachments/assets/227c2b09-93f0-421e-b9bc-5cfbd435c04a)

UCS:

![UCS](https://github.com/user-attachments/assets/dfb2d653-34bf-48ce-b423-bb1bdaa366f3)

IDS:

![IDS](https://github.com/user-attachments/assets/9a21ae2e-877c-4449-a2bc-630b2df27929)

So sánh hiệu suất các thuật toán tìm kiếm không có thông tin:

![image](https://github.com/user-attachments/assets/839e43ab-9c42-491f-bfea-d9802be279a4)


Nhận xét, BFS có số trạng thái duyệt qua ít hơn và thời gian duyệt nhanh hơn 3 thuật toán còn lại.
Cùng 1 trạng thái bắt đầu DFS mất thời gian quá dài do phải mở rộng quá nhiều trạng thái nên không đưa ra được số trạng thái đã mở rộng và thời gian hoàn thành (0,0).
IDS có độ phức tạp về thời gian nhỏ hơn DFS và độ phức tạp về không gian nhỏ hơn BFS.

### 2.2 Các thuật toán tìm kiếm có thông tin

Các thành phần chính bao gồm mảng hai chiều: trạng thái bắt đầu và kết thúc, và solution.
Các hàm để tính chi phí như hàm tính chi phí của đường đi  g(x), hàm dự đoán chi phí h(x) và hàm tổng hợp f(x) = g(x) + h(x).
Các cấu trúc lưu trữ trạng thái đã duyệt tương tự như không có thông tin.

Greedy Best-First Search:

![GBFS](https://github.com/user-attachments/assets/71b58ad9-1875-4501-8186-54e1808b4b3f)

A*:

![Astar](https://github.com/user-attachments/assets/0dd6a455-caa7-46de-a6a9-c8695b87ca2d)

IDA*:

![IDA](https://github.com/user-attachments/assets/763abe43-739f-4484-ab4d-d18deae7087a)

So sánh hiệu quả các thuật toán tìm kiếm có thông tin:

![image](https://github.com/user-attachments/assets/1d6c45e9-ddef-4abf-b085-75429df6369f)

Nhận xét: Tìm kiếm tham lam GBFS tuy có thời gian và số trạng thái duyệt qua là nhỏ nhất. Nhưng lại có số bước giải dài nhất. 
A* có bước giải, số trạng thái duyệt qua và thời gian ngắn hơn IDA* nhưng IDA* tối ưu hơn về không gian.

### 2.3 Các thuật toán tìm kiếm cục bộ.

Các thành phần chính bao gồm mảng hai chiều: trạng thái bắt đầu và kết thúc, và solution.

Trạng thái tốt nhất hiện tại, các thuật toán tìm kiếm cục bộ không lưu các trạng thái đã duyệt qua mà chỉ lưu trạng thái được đánh giá là tốt nhất.

Simple Hill Climbing: 

![Simple](https://github.com/user-attachments/assets/d70a587a-2e86-4769-a3ee-43dc6ac3d17d)

Steepest Ascent Hill Climbing:

![SA](https://github.com/user-attachments/assets/40506646-49c6-4899-aa90-3b9fb08125da)

Stochastic Hill Climbing:

![sto](https://github.com/user-attachments/assets/f30f7244-b93d-4a8e-8531-4481677944ca)

Simulated Annealing:

![sti](https://github.com/user-attachments/assets/75dbd937-b70e-456e-8d8c-b32b8d868059)

Beam:

![Beam](https://github.com/user-attachments/assets/13b453d6-886d-4722-a6dd-047132dea5fe)

Genetic Algorithms:

![Gener](https://github.com/user-attachments/assets/e4c57365-e3b0-423a-b213-ada07b0198aa)

So sánh hiệu quả các thuật toán tìm kiếm cục bộ:
**Các thuật toán tìm kiếm cục bộ có sự thiếu ổn định, không đảm bảo luôn đưa ra đáp án, đôi khi cùng một trạng thái bắt đầu nhưng khi chạy vài lần lại ra kết quả nên dùng biểu đồ như 2 loại trước lại thể hiện sự so sánh tương đối và không chính xác**
**Đôi khi lời giải giữa các lần chạy cũng không giống nhau**

Nhận xét: Tìm kiếm cục bộ có ưu điểm là không gian trạng thái hữu hạn, tốc độ chạy nhanh, nhưng phụ thuộc nhiêu vào cách xử lý vấn đề như kẹt cực trị cục bộ, phẳng, đỉnh giả, lặp vô nghĩa. Nhưng cho dù là vậy vẫn tồn tại khả năng không đưa ra đáp án.

### 2.4. Các thuật toán tìm kiếm trong môi trường niềm tin.
Đặc trưng của môi trường niềm tin chính là sự không chắc chắn.

And-Or Search: Không chắc chắn về kết qủa sau mỗi hành động

No observation: Không chắc chắn về môi trường

Partially observation: Thấy một phần của môi trường.

Có không gian trạng thái bắt đầu, kết thúc, các hành động và solution.

And-Or Search:

![and_or](https://github.com/user-attachments/assets/886e5475-ce9a-46f5-b81b-c8a189e3753e)

No observation:

![Belief](https://github.com/user-attachments/assets/e8ff227b-b856-481b-8eb8-7a59978ad83f)

Partially observation:

![Partially](https://github.com/user-attachments/assets/0a68260e-5965-4dc7-ae2b-fc4c32dff9c7)

So sánh hiệu suất thuật toán: 
And-Or search có hiệu suất chậm hơn một chút do dùng đệ quy, rồi tới No observation, và cuối cùng là Partially Observation.
Do No observation cần tìm lời giải có thể giải quyết tất cả trạng thái trong không gian, còn Partially chỉ mở rộng các trạng thái thoải với phần mà nó quan sát được.

Nhận xét: Tìm kiếm trong môi trường niềm tin là vấn đè không quá phù hợp khi dùng để giải bài toán 8 puzzle, tuy vẫn có thể đưa ra kết quả nhưng không phải trong mọi trường hợp và không tối ưu bằng các thuật toán khác.

### 2.5 Các thuật toán tìm kiếm trong môi trường ràng buộc.

Khác với các nhóm thuật toán trên, thuật toán trong môi trường ràng buộc không có trạng thái bắt đầu, chỉ có đưa ra kết quả phù hợp với các ràng buộc cho trước.

Kiểm thử: 

![Kiemthu](https://github.com/user-attachments/assets/044bc9d2-ad0a-48c9-bd1f-e7fe2db18cba)

Backtracking:

![backtrack](https://github.com/user-attachments/assets/872289a6-b56b-4659-8329-028763706932)

AC-3:

![ac3](https://github.com/user-attachments/assets/07773d36-f4dd-4add-bd9a-63142b441efd)

So sánh hiệu suất thuật toán tìm kiếm trong môi trường ràng buộc:
Kiểm thử có hiệu xuất thấp hơn 1 chút so với AC3 do phải duyệt qua nhiều trạng thái, nhưng với những bài toán đơn giản thì kiểm thử lại tỏ ra nhanh hơn, Backtracking trung bình.

Nhận xét: Tối ưu khi loại bỏ sớm các kết quả vi phạm ràng buộc. Tốn chi phí cao khi có quá nhiều ràng buộc ở các bài toán lớn.

### 2.6 Thuật toán học tăng cường

Có thêm một Q-table để lưu trữ các giá trị đánh giá chất lượng sau các hành động áp dụng lên 1 trạng thái.

Q- Learning:

![Q2](https://github.com/user-attachments/assets/019c0d4d-377b-45cb-a991-9f8a9fad57e8)

So sánh hiệu suất: Tốc độ không cao do phải trải qua quá trình train. Đôi khi vẫn sẽ bị vướng kẹt đỉnh cục bộ do cách tính phần thưởng(em dùng khoảng cách manhattan nên dễ kẹt).

Nhận xét: Q - Learning là thuật toán học tăng cường cơ bản, có thể đáp ứng khi giải các bài toán đơn giản, với các bài toán phức tạp hơn như không gian trạng thái lớn, hành động nhiều, yêu cầu phức tạp, yêu cầu tính tổng quát thì không nên dùng Q - Learning

# 3. Kết luận.
