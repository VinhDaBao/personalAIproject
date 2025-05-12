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
