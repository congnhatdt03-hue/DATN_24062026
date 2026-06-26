# GUI_STEP_DEBUG_SPEC.md

Tai lieu nay mo ta GUI moi cho `detect_stator`, tap trung vao kha nang debug tung buoc thay vi chi chay terminal.

## Muc tieu giao dien

- Giao dien dep, de nhin, uu tien ro rang va thuan tien cho debug.
- Co sidebar hoac tab cho 7 buoc.
- Bam buoc nao hien panel buoc do.
- Moi buoc co nut Run rieng.
- Khi bam Run o buoc nao thi ket qua buoc do phai hien ngay tren GUI.
- Co checkbox Auto Update.
- Co nut Save Preset, Load Preset, Reset.
- Co vung hien thi anh lon.
- Co combobox chon anh debug.
- Co combobox chon ROI ID neu buoc do dung ROI.
- Co vung log.
- Co bang ket qua o buoc tong hop.

## Nguyen tac interactive

- Chinh tham so -> bam Run -> anh cap nhat ngay.
- Neu Auto Update bat, chinh tham so thi tu chay lai sau 300-500 ms.
- Khong chi in terminal.
- Khong chi luu anh ra output roi bat nguoi dung mo thu cong.
- GUI chi goi cac ham trong `src/`.
- Moi panel can hien du anh trung gian va thong tin log de debug nhanh.

## Kien truc panel de xuat

1. `HoughStepPanel`
2. `RoiStepPanel`
3. `TabEdgeStepPanel`
4. `RadialStepPanel`
5. `TemplateStepPanel`
6. `MatchingStepPanel`
7. `FullPipelinePanel`

## Bo cuc tong the

### Cot trai

- Sidebar hoac `Notebook` de chon 7 buoc.
- Nhom action chung:
  - load anh
  - load preset theo buoc
  - luu preset theo buoc
  - reset tham so

### Cot giua

- Vung hien thi anh chinh.
- Ho tro zoom fit / pan neu can.
- Cho phep doi anh debug bang combobox.

### Cot phai hoac phia duoi

- Khu tham so cua buoc hien tai.
- Vung log.
- Vung thong tin tom tat ket qua.

## Yeu cau cho moi panel

### HoughStepPanel

- Tham so tien xu ly va Hough duoc nhom ro.
- Hien 4 anh debug: goc, tien xu ly, candidates, filtered.
- Log so luong circle truoc va sau loc.
- Bang ket qua circle nen co them `support_pct` de nguoi dung giam sat muc do khop hien tai cua moi vong tron.

### RoiStepPanel

- Chon ROI theo ID.
- Hien anh tong quan co khung ROI va ROI chi tiet.
- Co them che do xem ROI da ve circle refine va danh dau tam.
- Co bang ket qua `ID`, `center_x`, `center_y`, `radius`, `score`.
- Cac tham so ROI refine co slider va co the luu preset JSON.
- Cac tham so noi bo nhu `radius mask` va trong so `score` duoc giu o backend, khong dua len GUI ROI de tranh roi giao dien.
- Khi tune o tab ROI, chi ROI ID dang chon duoc refine; cac ROI con lai chi cat va luu de giam lag.

### TabEdgeStepPanel

- Hien day du chuoi debug:
  `roi_gray`, `roi_preprocessed`, `canny_edges`, `radius_mask`, `tab_edges_raw`, `tab_edges_clean`, `debug_overlay`.
- Cac tham so loc radius va connected component phai chinh truc tiep tren panel.

### RadialStepPanel

- Hien anh ray overlay.
- Hien do thi signature.
- Cho phep chon ROI ID neu dang debug tu anh khay.

### TemplateStepPanel

- Tao va luu template 0 do.
- Xem anh debug va metadata cua template.
- Co them mot mode `Anh debug` de xem ROI goc kem truc toa do XY, tam dat tai tam anh va truc keo dai gan het khung hinh.

### MatchingStepPanel

- Hien ROI test, `tab_edges`, radial rays, va do thi MSE.
- Hien thong tin mau o mot dong tom tat in dam ben ngoai bang ket qua.
- Bang ket qua co `goc_tho_deg`, `goc_tinh_deg`, va `min_error`.
- Cac gia tri goc hien tren GUI nen dung quy uoc signed `-180 .. 180` de de doc cho nguoi van hanh.

### FullPipelinePanel

- Hien `final_result`.
- Hien bang ket qua tong hop.
- Hien log tung stator va canh bao status.

## Du lieu GUI can quan ly

- Duong dan anh khay hien tai.
- Duong dan ROI mau hoac template.
- Preset dang duoc load theo tung buoc.
- Ket qua trung gian theo ID.
- Lua chon ROI hien tai.
- Trang thai Auto Update.

## Nguyen tac trien khai sau nay

- Khong dua code xu ly anh vao callback GUI.
- Moi su kien GUI chi dong vai tro thu tham so va goi service trong `src/`.
- Ket qua xu ly nen tra ve theo dang `success/data/images/logs` de GUI hien thi dong nhat.
