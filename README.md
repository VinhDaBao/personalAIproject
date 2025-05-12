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
