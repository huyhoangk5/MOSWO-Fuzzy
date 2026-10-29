# MOSWO-Fuzzy: Multiobjective Spider-Wasp Optimizer with Fuzzy Adaptive Control

Chương trình cài đặt thuật toán **Multiobjective Spider-Wasp Optimizer (MOSWO)** tích hợp **Bộ điều khiển mờ thích ứng Mamdani (4 đầu vào – 6 đầu ra)** cùng các cơ chế thoát bẫy cực trị địa phương (**Đột biến Cauchy**, **Tái khởi tạo một phần thích ứng – Partial Reinitialization**, và **Học dựa trên đối lập – Opposition-Based Learning**).

---

## 1. Kiến trúc Thuật toán

### Bộ điều khiển mờ thích ứng (4 Inputs – 6 Outputs)
* **4 Đầu vào (Antecedents):**
  1. `iter_ratio` ($t / T \in [0, 1]$): Tiến trình thế hệ hiện tại.
  2. `diversity` ($\in [0, 1]$): Độ đa dạng của quần thể lưu trữ (Archive) trong không gian mục tiêu chuẩn hóa.
  3. `g_excess` ($\in [0, 1]$): Độ lệch ước lượng khoảng cách từ quần thể hiện tại tới mặt Pareto lý tưởng (nhận diện trạng thái hội tụ thực sự hay bị kẹt tại cực trị địa phương của hàm $g$).
  4. `dec_diversity` ($\in [0, 1]$): Độ phân tán của các biến quyết định trong không gian tìm kiếm.
* **6 Đầu ra (Consequents):**
  1. `mu_scale` ($\in [0.1, 2.0]$): Hệ số điều chỉnh biên độ bước nhảy khám phá (Exploration).
  2. `c_factor` ($\in [0.1, 1.0]$): Hệ số điều chỉnh tốc độ bám theo cá thể dẫn đầu (Exploitation).
  3. `reinit_strength` ($\in [0.0, 1.0]$): Tỷ lệ tái khởi tạo một phần các chiều biến kiểm soát hội tụ ($g$-dimensions) khi quần thể bị trì trệ.
  4. `cr_rate` ($\in [0.4, 0.98]$): Tốc độ lai ghép thích ứng (Crossover rate).
  5. `eta_m` ($\in [2.0, 25.0]$): Chỉ số phân phối của đột biến đa thức (Polynomial mutation index).
  6. `cauchy_prob` ($\in [0.0, 0.25]$): Xác suất kích hoạt bước nhảy đột biến phân phối nặng đuôi Cauchy.

---

## 2. Cấu trúc Thư mục

| File | Mô tả |
| :--- | :--- |
| `MOSWO.py` | Cài đặt lớp thuật toán `MOSWO` chính và các cấu hình biến thể. |
| `fuzzy_controller_skfuzzy.py` | Cài đặt bộ điều khiển mờ Mamdani 4 đầu vào – 6 đầu ra (`FuzzyMOSWOController`). |
| `benchmarks.py` | Định nghĩa 12 bài toán kiểm thử chuẩn: `ZDT1`–`ZDT4`, `ZDT6` (2 mục tiêu) và `DTLZ1`–`DTLZ7` (3 mục tiêu). |
| `main.py` | Chương trình chạy đánh giá thuật toán trên bộ benchmark với các chỉ số **IGD**, **HV**, **Spread** và thời gian chạy. |
| `ablation_study.py` | Chương trình chạy thực nghiệm cắt bỏ (Ablation Study) so sánh 7 biến thể từ nguyên bản đến mô hình đầy đủ. |
| `requirements.txt` | Danh sách các thư viện Python cần thiết cho môi trường thực thi. |

---

## 3. Cài đặt Môi trường

Yêu cầu **Python 3.10+** (khuyến nghị tạo môi trường ảo `.venv`):

```powershell
# 1. Tạo môi trường ảo (nếu chưa có)
python -m venv .venv

# 2. Kích hoạt môi trường ảo trên Windows PowerShell
.\.venv\Scripts\Activate.ps1

# 3. Cài đặt các thư viện từ requirements.txt
pip install -r requirements.txt
```

---

## 4. Hướng dẫn Chạy Chương trình

### 4.1. Chạy kiểm thử Benchmark tiêu chuẩn (`main.py`)
Chạy đánh giá mô hình đầy đủ trên toàn bộ 12 bài toán chuẩn ZDT và DTLZ:

```powershell
.\.venv\Scripts\python.exe main.py
```

### 4.2. Chạy Thực nghiệm Cắt bỏ – Ablation Study (`ablation_study.py`)
Đánh giá và so sánh 7 biến thể (`MOSWO`, `MOSWO-V`, `MOSWO-A`, `MOSWO-F`, `MOSWO-F+Cauchy`, `MOSWO-F+Reinit`, `MOSWO-Full`) kèm kiểm định thống kê Wilcoxon:

```powershell
# Chạy toàn bộ 12 bài toán (mặc định 10 runs, có thể tùy chỉnh qua --runs)
.\.venv\Scripts\python.exe ablation_study.py --runs 10

# Hoặc chỉ chạy trên một số bài toán cụ thể:
.\.venv\Scripts\python.exe ablation_study.py --runs 5 --problems ZDT1 ZDT4 DTLZ1 DTLZ3
```
