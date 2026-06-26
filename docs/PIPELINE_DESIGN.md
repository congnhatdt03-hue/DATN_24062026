# PIPELINE_DESIGN.md

Tai lieu nay mo ta pipeline 7 buoc cua project `detect_stator`.
Muc tieu la chuan hoa logic xu ly, xac dinh ro dau vao - dau ra, va dinh nghia GUI debug cho tung buoc.

## Buoc 1: HoughCircle

### Muc tieu

Tu anh khay tim tam va ban kinh 12 stator.

### GUI can co

- Nut chon anh.
- Checkbox bat/tat tien xu ly.
- Nhom tham so tien xu ly:
  - `use_clahe`
  - `clahe_clip_limit`
  - `clahe_tile_grid_size`
  - `use_gaussian`
  - `gaussian_kernel`
- Nhom tham so Hough:
  - `hough_dp`
  - `hough_param1`
  - `hough_param2`
  - `hough_minDist`
  - `hough_minRadius`
  - `hough_maxRadius`
  - `min_center_dist`
  - `expected_count`
- Checkbox Auto Update.
- Nhom `Fast Mode`:
  - `fast_mode.enabled`
  - `fast_mode.max_processing_dim`
- Nut Run.
- Nut Save Preset.
- Nut Load Preset.
- Vung anh hien thi:
  - anh goc
  - anh tien xu ly
  - Hough candidates
  - Hough filtered
- Log:
  - so circle truoc loc
  - so circle sau loc
  - canh bao neu khac 12

### Dau ra

- `circles`: `list[dict]` voi cau truc `{id, x, y, r, score, support_pct}`.
- Bang tab 1 nen hien them `support_pct` de theo doi muc do edge coverage cua tung vong tron Hough.
- `hough_preset.json`.
- Anh debug cua buoc 1.

### Ghi chu thiet ke

- Fast mode chi duoc dung de detect Hough nhanh hon:
  - Resize anh khay xuong nho hon de detect.
  - Scale `x`, `y`, `r` ve he toa do anh goc.
  - Cac buoc ROI, tab edge, radial, va matching van dung ROI cat tu anh goc.
- GUI buoc 1 nen chay detect trong `threading.Thread`, day ket qua qua `queue.Queue`, va main thread poll bang `root.after` de tranh dung giao dien.

## Buoc 2: Cat ROI

### Muc tieu

Cat ROI tung stator tu ket qua Hough, sau do chay Hough tinh lai trong tung ROI de chot tam, ban kinh, va score tot hon truoc khi sang buoc loc bien tai.

### Dau vao

- Anh khay.
- `circles` tu buoc 1.
- `roi_preset`.

### GUI can co

- Nut load anh.
- Nut load Hough preset.
- Nut Run.
- Combobox chon ID ROI.
- Tham so:
  - `roi_scale` hoac `half_size_scale`
  - `output_size` neu can
  - nhom tham so `Hough refine` / `Preprocess` / `Canny` / `Radius mask` / `Score`
- Vung anh:
  - anh toan canh co khung ROI
- ROI tung ID
  - ROI co ve circle refine va danh dau tam
- Bang ket qua `ID`, `center_x`, `center_y`, `radius`, `score`.
- Nut Save ROI.

### Cach chay de tune

- Khi bam `Run` o buoc nay:
  - cat ROI cho tat ca stator tu anh goc
  - luu ROI ra `data/roi/`
  - chi refine Hough tren ROI ID dang chon de tune cho muot
- Khi chay `Full Pipeline`:
  - refine Hough cho tat ca ROI bang preset da luu

### Dau ra

- `rois`: `dict[id] -> roi_image`.
- `roi_metadata`: `center_in_roi`, `radius`, `offset_x`, `offset_y`, `center_x`, `center_y`, `radius_full`, `score`.
- Anh ROI luu trong `data/roi/`.

### Ghi chu thiet ke

- Uu tien tan dung logic cat ROI tu file `2. cat roi.py`.
- Buoc nay la noi giao tiep giua phat hien tron va xu ly theo tung stator.

## Buoc 3: Loc bien tai stator

### Muc tieu

Tu ROI tao anh `tab_edges_clean` chi giu bien tai stator.

### Luong xu ly

`ROI`
-> `YOLO detect box tai`
-> `crop tung tai + padding`
-> `grayscale`
-> `CLAHE` neu bat
-> `blur`
-> `Canny`
-> `morph close`
-> `external contours`
-> `loc contour theo dien tich + khoang cach so voi tam`
-> `outer profile contour`
-> `tab_edges_raw`
-> `tab_edges_clean`
-> `debug overlay`

### Phuong phap chinh

- Dung YOLO de khoanh vung tung tai tren ROI stator.
- Trong moi crop tai, dung `CLAHE + blur + Canny + morphology close`.
- Chi giu contour ngoai hop le, sau do lay `outer profile` nhin tu tam stator.
- Dau ra cuoi cung van la anh `tab_edges_clean` de phuc vu Radial Signature.

### Dieu kien loc contour

- `area >= min_area`
- `area >= crop_area * min_area_ratio`
- `contour_max_distance >= detection_center_distance * min_keep_distance_ratio`
- Moi bin goc chi giu diem xa tam nhat de tao `outer profile`

### GUI can hien thi

- ROI Gray
- Preprocessed
- Canny Edges
- YOLO Boxes
- Closed Edges
- Tab Edges Raw
- Tab Edges Clean
- Debug Overlay

### Dau ra

- `tab_edges_clean`
- `tab_edge_preset.json`

## Buoc 4: Radial Signature

### Muc tieu

Tao signature tu `tab_edges_clean`.

### Cach lam

- Lay cac diem bien cua `tab_edges_clean`.
- Voi moi diem bien tinh `rho` va `angle`.
- Chia `angle` thanh 360 bin.
- Moi bin lay `rho` lon nhat.
- Noi suy khoang trong ngan neu can.
- Chuan hoa signature.
- Ve tia tu tam den diem bien.

### GUI can hien thi

- anh radial rays
- signature plot
- bang 360 gia tri neu can

### Dau ra

- `signature`
- `radial_debug`
- thong tin tam va ban kinh su dung khi quet

## Buoc 5: Tao mau 0 do

### Muc tieu

Tao du lieu mau de so khop goc.

### Dau vao

- ROI mau 0 do hoac anh mau 0 do.

### Dau ra

- `template_signature.npy` hoac `template_signature.json`
- `template_data.json` gom:
  - `signature`
  - `center`
  - `radius`
  - `params_used`
  - `note: 0 degree template`
- Anh debug:
  - `template_tab_edges.png`
  - `template_radial_rays.png`
  - `template_signature.png`
  - them mot mode xem ROI goc co overlay truc toa do XY, tam trung voi tam anh va truc keo dai qua bien stator.

## Buoc 6: So khop MSE

### Muc tieu

Tinh goc lech giua ROI test va mau.

### Cach lam

- Tao signature ROI test.
- Load `template_signature`.
- Dich vong signature tu 0 den 359 do.
- Tinh MSE tung goc.
- `goc_tho_deg` la goc discrete ung voi chi so MSE nho nhat.
- Noi suy parabol quanh cuc tieu de tinh `goc_tinh_deg`.
- `angle_deg` la gia tri goc tinh cuoi cung de giu compatibility cho cac buoc sau.

### GUI can hien thi

- ROI test
- tab_edges test
- radial rays test
- do thi MSE
- thong tin mau in dam ben ngoai bang ket qua
- `goc_tho_deg`
- `goc_tinh_deg`
- `min_error`
- Tren man hinh, cac goc hien thi theo thang signed `-180 .. 180`; du lieu noi bo van co the giu quy uoc `0 .. 360` de phuc vu tinh toan / compatibility.

### Dau ra

- `match_result`
- `mse_curve`
- `best_angle`
- `goc_tho_deg`
- `goc_tinh_deg`

## Buoc 7: Tong hop

### Muc tieu

Gop tat ca cac buoc thanh chuong trinh hoan chinh.

### Dau vao

- Anh khay.
- `hough_preset.json`
- `roi_preset.json`
- `tab_edge_preset.json`
- `template_data.json`

### Dau ra

- `final_result.png`
- `results.csv`
- Bang ket qua:

| ID | center_x | center_y | radius | angle_deg | min_error | status |
| -- | -------- | -------- | ------ | --------- | --------- | ------ |

### GUI can hien thi

- anh `final_result`
- bang ket qua
- log
- danh sach ROI / `tab_edges` theo ID

### Ghi chu thiet ke

- Neu Hough dang bat fast mode thi tong hop van phai di theo luong:
  - Detect Hough tren anh resize.
  - Scale circle ve anh goc.
  - Cat ROI tren anh goc.
  - Tien xu ly va tinh goc tren ROI anh goc da cat.

## Nguyen tac xuyen suot

- Moi buoc phai chay doc lap duoc.
- Moi buoc phai tra anh debug ngay trong GUI.
- Preset moi buoc phai luu duoc de tai su dung.
- Khong viet thuat toan xu ly anh truc tiep trong GUI.
- Hien tai buoc 3 duoc phep dung YOLO de khoanh vung tai stator, nhung dau ra van phai la `tab_edges_clean` phuc vu Radial Signature.
