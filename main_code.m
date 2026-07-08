clc; clear; close all; warning('off', 'all');

fprintf('╔══════════════════════════════════════════╗\n');
fprintf('║   KABLO ANALİZ SİSTEMİ                                                                        \n');
fprintf('║   YOLOv11 + L*a*b* + GLCM Doku Analizi                                                \n');
fprintf('╚══════════════════════════════════════════╝\n\n');

%% ─── 1. GÖRÜNTÜ SEÇİMİ ────────────────────────────────────────────────────
[dosya, yol] = uigetfile( ...
    {'*.jpg;*.jpeg;*.png;*.bmp;*.tif;*.tiff', 'Görüntü Dosyaları'}, ...
    'Analiz Edilecek Kablo Görüntüsünü Seçin');
if isequal(dosya, 0)
    error('Dosya seçilmedi. Program sonlandırılıyor.');
end
orijinal = imread(fullfile(yol, dosya));
[H, W, ~] = size(orijinal);
px_mm = 0.0572;   
fprintf('[✓] Görüntü : %s  (%d × %d px)\n', dosya, W, H);

%% ─── 2. ÖN İŞLEME ─────────────────────────────────────────────────────────
if size(orijinal, 3) == 3
    gri = rgb2gray(orijinal);
else
    gri = orijinal;
end
gri = uint8(gri);
gri_eq = adapthisteq(gri, 'ClipLimit', 0.02, 'NumTiles', [8 8]);
sigma = max(0.8, min(H,W) * 0.004);
g_filtre = imgaussfilt(double(gri_eq), sigma);
fprintf('[✓] Ön işleme: CLAHE + Gaussian (σ=%.2f)\n', sigma);

%% ─── 3. KABLO SEGMENTASYONU (ZORLU ZEMİN VE AHŞAP DOKU KORUMASI) ──────────────────
if size(orijinal, 3) == 3
    lab_img = rgb2lab(orijinal);
    L_ch = imgaussfilt(lab_img(:,:,1), sigma * 3); 
    A_ch = lab_img(:,:,2); B_ch = lab_img(:,:,3);
    ust_kenar = L_ch(1:10, :); alt_kenar = L_ch(end-9:end, :);
    sol_kenar = L_ch(11:end-10, 1:10); sag_kenar = L_ch(11:end-10, end-9:end);
    tum_kenarlar = [ust_kenar(:); alt_kenar(:); sol_kenar(:); sag_kenar(:)];
    bg_renk = median(tum_kenarlar, 'omitnan');
    T_otsu = graythresh(L_ch/100) * 100;
    bw_A = (L_ch < T_otsu); 
    if bg_renk < T_otsu - 5
        bw_A = (L_ch > T_otsu); 
    end
    renk_mag = sqrt(A_ch.^2 + B_ch.^2);
    bw_renk = renk_mag > 15; 
    bw_A = bw_A | bw_renk; 
else
    ust_kenar = g_filtre(1:10, :); alt_kenar = g_filtre(end-9:end, :);
    sol_kenar = g_filtre(11:end-10, 1:10); sag_kenar = g_filtre(11:end-10, end-9:end);
    tum_kenarlar = double([ust_kenar(:); alt_kenar(:); sol_kenar(:); sag_kenar(:)]);
    bg_renk = median(tum_kenarlar, 'omitnan');
    T_otsu = graythresh(uint8(g_filtre));
    bw_A = imbinarize(uint8(g_filtre), T_otsu);
    if bg_renk > (T_otsu * 255) - 10
        bw_A = ~bw_A;
    end
end
doku_bastirilmis = medfilt2(gri_eq, [13 13]); 
etkili_kenar  = edge(doku_bastirilmis, 'canny', [0.08 0.20]);
etkili_kenar  = bwareaopen(etkili_kenar, 300); % Ufak parazitleri yoksay
bw_C   = imfill(etkili_kenar, 'holes');
bw_raw = bw_A | bw_C;
r_tras = max(3, round(min(H,W) * 0.005)); 
bw_raw = imopen(bw_raw, strel('disk', r_tras)); 
r_kapat = max(5, round(min(H,W) * 0.015)); 
r_ac    = max(2, round(r_kapat / 4));      
bw_kal  = imclose(bw_raw, strel('disk', r_kapat));
bw_kal  = imopen (bw_kal, strel('disk', r_ac));
bw_fill   = imfill(bw_kal, 'holes');
holes_all = bw_fill & ~bw_kal;               
cc_holes  = bwconncomp(holes_all);
p_holes   = regionprops(cc_holes, 'Area');
max_delik = sum(bw_kal(:)) * 0.02; 
bw_kal2 = bw_kal;
for hi = 1:cc_holes.NumObjects
    if p_holes(hi).Area < max_delik          
        bw_kal2(cc_holes.PixelIdxList{hi}) = true;
    end
end
cc = bwconncomp(bw_kal2);
if cc.NumObjects == 0
    error('Kablo tespit edilemedi. Farklı bir görüntü deneyin.');
end
props_seg = regionprops(cc, 'Area','Orientation');
maske = false(H, W);
alan_esigi = round((H * W) * 0.002); 
tutulan_nesne_sayisi = 0;
for i = 1:cc.NumObjects
    if props_seg(i).Area > alan_esigi
        maske(cc.PixelIdxList{i}) = true;
        tutulan_nesne_sayisi = tutulan_nesne_sayisi + 1;
    end
end
[~, idx_max] = max([props_seg.Area]);
if ~isempty(idx_max)
    aci = props_seg(idx_max).Orientation;
else
    aci = 0;
end
fprintf('[✓] Segmentasyon L*a*b* Desteği İle Tamamlandı | Açı: %.1f°\n\n', aci);

%% ─── 4. YAPAY ZEKA TEŞHİSİ (ÇAP ÖLÇÜMÜNDEN ÖNCEYE ALINDI) ─────────────────
fprintf('╔─ YAPAY ZEKA TEŞHİSİ (YOLOv11) ─────────╗\n');
fprintf('║ Python köprüsü tetikleniyor...                                           \n');
resim_tam_yol = fullfile(yol, dosya);
python_yolu = '"C:\Users\Suat\AppData\Local\Python\pythoncore-3.14-64\python.exe"';
komut = sprintf('%s yolo_kopru.py "%s"', python_yolu, resim_tam_yol);
[status, cmdout] = system(komut); 
disp(strtrim(cmdout));
yolo_sinif = 'Bilinmiyor';
yolo_guven = 0;
num_yolo_boxes = 0;
yolo_bboxes = [];
if isfile('yolo_sonuc.txt')
    fid = fopen('yolo_sonuc.txt', 'r');
    sonuc_str = fgetl(fid);
    fclose(fid);
    if ischar(sonuc_str)
        parcalar = strsplit(sonuc_str, ',');
        if length(parcalar) >= 3
            yolo_sinif = parcalar{1};
            yolo_guven = str2double(parcalar{2});
            num_yolo_boxes = str2double(parcalar{3});
            if num_yolo_boxes > 0 && length(parcalar) >= (3 + num_yolo_boxes * 4)
                yolo_bboxes = zeros(num_yolo_boxes, 4);
                idx_p = 4;
                for b = 1:num_yolo_boxes
                    x1 = str2double(parcalar{idx_p});
                    y1 = str2double(parcalar{idx_p+1});
                    x2 = str2double(parcalar{idx_p+2});
                    y2 = str2double(parcalar{idx_p+3});
                    yolo_bboxes(b, :) = [x1, y1, (x2-x1), (y2-y1)];
                    idx_p = idx_p + 4;
                end
            end
        end
    end
end
fprintf('║ Teşhis : %-20s                                                                                         \n', yolo_sinif);
fprintf('║ Güven  : %%%-6.2f                                                                                 \n', yolo_guven);
fprintf('╚─────────────────────────────────────────╝\n\n');
esik_degeri = 40.0; 
if strcmp(yolo_sinif, 'SAĞLAM') || yolo_guven < esik_degeri
    yolo_risk_str = 'SAĞLAM (YOLO)';
    yolo_risk_renk = [0.1 0.85 0.2]; 
    yolo_sinif = 'SAĞLAM'; 
else
    yolo_risk_str = sprintf('HASARLI (%s)', strrep(upper(yolo_sinif), '_', ' '));
    yolo_risk_renk = [1.0 0.15 0.15]; 
end

%% ─── 5. ÇAP ÖLÇÜMÜ (İSKELET BUDAMA VE YOLO KILAVUZLU) ────────────────────
dt_maske = bwdist(~maske);
iskelet = bwmorph(maske, 'skel', Inf); 
iskelet = bwmorph(iskelet, 'spur', 15);
[isk_r, isk_c] = find(iskelet);
gecerli_maske = true(size(isk_r));
if ~strcmp(yolo_sinif, 'SAĞLAM') && num_yolo_boxes > 0
    for b = 1:num_yolo_boxes
        bx = yolo_bboxes(b, 1);
        by = yolo_bboxes(b, 2);
        bw = yolo_bboxes(b, 3); bh = yolo_bboxes(b, 4);
        tampon = 10; 
        in_box = (isk_c >= bx-tampon) & (isk_c <= bx+bw+tampon) & ...
                 (isk_r >= by-tampon) & (isk_r <= by+bh+tampon);
        gecerli_maske(in_box) = false; 
    end
end
yaricaplar = dt_maske(iskelet);
gecerli_yari = yaricaplar(gecerli_maske);
gecerli_yari = gecerli_yari(gecerli_yari > 0);
if numel(gecerli_yari) >= 5
    q1 = quantile(gecerli_yari, 0.25); 
    q3 = quantile(gecerli_yari, 0.75);
    iqr_v = q3 - q1;
    temiz_yari = gecerli_yari(gecerli_yari >= (q1-1.5*iqr_v) & gecerli_yari <= (q3+1.5*iqr_v));
    cap_px = 2 * mean(temiz_yari, 'omitnan');
else
    cap_px = 2 * median(gecerli_yari, 'omitnan'); 
end
cap_mm = cap_px * px_mm;
fprintf('╔─ ÇAP ANALİZİ (Akıllı İskelet Yöntemi) ─╗\n');
fprintf('║ Kablo Çapı       : %6.2f piksel                                                                  \n', cap_px);
fprintf('║ Kablo Çapı       : %6.2f mm                                                                     \n', cap_mm);
fprintf('╚─────────────────────────────────────────╝\n\n');

%% ─── 6. MATEMATİKSEL HASAR KONUMU TESPİTİ (GLCM DOKU ANALİZİ İLE) ────────
kenar_trasla = double(max(1, round(cap_px * 0.15)));
maske_ic = imerode(maske, strel('disk', kenar_trasla));
if sum(maske_ic(:)) < 10, maske_ic = maske; end
px_kablo = double(gri(maske_ic));
mu_k  = mean(px_kablo, 'omitnan');
sg_k  = std(px_kablo, 'omitnan');
p90_k = quantile(px_kablo, 0.90);
if isnan(p90_k), p90_k = 200; end 
esik_b = max(220, p90_k + 2.0 * sg_k); 
hasar_b = (double(gri) > esik_b) & maske_ic;
win_s = max(5, round(cap_px * 0.15)); 
if mod(win_s,2)==0, win_s = win_s+1; end
glcm_entropy = entropyfilt(uint8(gri), ones(win_s));
entropy_kablo = glcm_entropy(maske_ic);
esik_entropy = quantile(entropy_kablo, 0.98); 
if isnan(esik_entropy), esik_entropy = 10; end 
hasar_doku = (glcm_entropy > esik_entropy * 1.5) & maske_ic;   
hasar_c = false(H, W);
if size(orijinal,3) == 3
    hsv_img = rgb2hsv(orijinal);
    H_ch = hsv_img(:,:,1); S_ch = hsv_img(:,:,2); V_ch = hsv_img(:,:,3);
    hasar_sicak = ((H_ch < 0.12 | H_ch > 0.85) & S_ch > 0.25 & V_ch > 0.08) & maske_ic;
    hasar_koyu = (double(gri) < (mu_k - 3.0 * sg_k)) & maske_ic; 
    hasar_c = hasar_sicak | hasar_koyu;
end
oy = double(hasar_b) + double(hasar_doku) + double(hasar_c);
hasar_raw = (oy >= 2) | hasar_c;
ortalama_yaricap = cap_px / 2;
hata_toleransi = ortalama_yaricap * 0.45; 
hatali_iskelet = iskelet & (abs(dt_maske - ortalama_yaricap) > hata_toleransi);
geo_yaricap = double(max(1, round(ortalama_yaricap)));
hasar_geometrik = imdilate(hatali_iskelet, strel('disk', geo_yaricap));
hasar_geometrik = hasar_geometrik & maske; 
hasar_raw = hasar_raw | hasar_geometrik;
kenar_marji = max(15, round(cap_px * 1.5)); 
sinir_maskesi = false(H, W);
y_bas = min(H, kenar_marji); y_bit = max(1, H-kenar_marji);
x_bas = min(W, kenar_marji); x_bit = max(1, W-kenar_marji);
sinir_maskesi(y_bas:y_bit, x_bas:x_bit) = true;
hasar_raw = hasar_raw & sinir_maskesi;
min_alan = double(round(max(25, cap_px^2 * 0.05))); 
max_alan = double(round(max(min_alan + 1, sum(maske(:)) * 0.08))); 
hasar_temiz_ilk = bwareaopen(hasar_raw, min_alan);
dev_hasarlar = bwareaopen(hasar_temiz_ilk, max_alan);
hasar_temiz = hasar_temiz_ilk & ~dev_hasarlar;
hasar_temiz = imdilate(hasar_temiz, strel('disk', 3));
hasar_temiz = hasar_temiz & maske;
[hasar_etiket, h_sayi] = bwlabel(hasar_temiz);

%% ─── 7. GÖRSELLEŞTİRME (HİBRİT PANO) ──────────────────────────────────────
fig = figure('Name','Kablo Analiz Sistemi','NumberTitle','off', ...
    'Position',[30 30 1500 820], 'Color',[0.07 0.07 0.10]);
ax1 = subplot(2,3,1); imshow(orijinal); title('① Orijinal Görüntü','Color','w','FontSize',11,'FontWeight','bold');
ax2 = subplot(2,3,2); imshow(gri_eq); title('② CLAHE Normalizasyon','Color','w','FontSize',11,'FontWeight','bold');
ax3 = subplot(2,3,3); imshow(etkili_kenar); title('③ Etkili Kenar Tespiti','Color','w','FontSize',11,'FontWeight','bold');
ax4 = subplot(2,3,4); imshow(maske);
title(sprintf('④ L*a*b* Segmentasyon | Açı: %.1f°', aci), 'Color','w','FontSize',11,'FontWeight','bold');
ax5 = subplot(2,3,5);
imshow(gri); hold on;
kablo_sinir = bwperim(maske);
[r_sinir, c_sinir] = find(kablo_sinir);
plot(c_sinir, r_sinir, 'g.', 'MarkerSize', 1);
plot(isk_c(~gecerli_maske), isk_r(~gecerli_maske), 'r.', 'MarkerSize', 3);
plot(isk_c(gecerli_maske), isk_r(gecerli_maske), 'y.', 'MarkerSize', 3);
hedef_yari = cap_px / 2; 
isk_r_gec = isk_r(gecerli_maske);
isk_c_gec = isk_c(gecerli_maske);
yari_cap_gec = yaricaplar(gecerli_maske);
if ~isempty(yari_cap_gec)
    [~, saglam_idx] = min(abs(yari_cap_gec - hedef_yari));
    saglam_r = isk_r_gec(saglam_idx);
    saglam_c = isk_c_gec(saglam_idx);
    yari_px = cap_px / 2;
    if abs(aci) <= 45 
        r_ust = max(1, saglam_r - yari_px);
        r_alt = min(H, saglam_r + yari_px);
        plot([saglam_c saglam_c], [r_ust r_alt], 'm-', 'LineWidth', 2.5); 
        plot([saglam_c-12 saglam_c+12], [r_ust r_ust], 'm-', 'LineWidth', 2.5);
        plot([saglam_c-12 saglam_c+12], [r_alt r_alt], 'm-', 'LineWidth', 2.5);
    else 
        c_sol = max(1, saglam_c - yari_px);
        c_sag = min(W, saglam_c + yari_px);
        plot([c_sol c_sag], [saglam_r saglam_r], 'm-', 'LineWidth', 2.5); 
        plot([c_sol c_sol], [saglam_r-12 saglam_r+12], 'm-', 'LineWidth', 2.5);
        plot([c_sag c_sag], [saglam_r-12 saglam_r+12], 'm-', 'LineWidth', 2.5);
    end
end
title(sprintf('⑤ Çap: %.2f mm', cap_mm), 'Color','w','FontSize',11,'FontWeight','bold'); hold off;
ax6 = subplot(2,3,6); imshow(hasar_temiz); title(sprintf('⑥ GLCM Doku Haritası (%d bölge)', h_sayi), 'Color','w','FontSize',11,'FontWeight','bold');
fig_detay = figure('Name', 'Detaylı Hasar Analizi', 'NumberTitle', 'off', ...
                   'WindowState', 'maximized', 'Color', [0.07 0.07 0.10]);
ax_detay = axes(fig_detay);
imshow(orijinal, 'Parent', ax_detay); hold on;
if strcmp(yolo_sinif, 'SAĞLAM')
    text(W/2, H/2, 'KABLO SAĞLAM', 'Color', 'g', 'FontSize', 18, ...
        'FontWeight', 'bold', 'HorizontalAlignment', 'center', 'BackgroundColor', [0 0 0 0.7]);
else
    for b = 1:num_yolo_boxes
        bbox = yolo_bboxes(b, :);
        rectangle('Position', bbox, 'EdgeColor', [1 0 0], 'LineWidth', 4, 'LineStyle', '-');
        text(bbox(1) + 3, bbox(2) - 3, sprintf(' HASAR %d (YOLO) ', b), ...
            'Color', 'yellow', 'FontSize', 14, 'FontWeight', 'bold', ...
            'BackgroundColor', [0.9 0.1 0.1], 'VerticalAlignment', 'bottom');
        if abs(aci) <= 45 
            hasar_boyu_mm = bbox(3) * px_mm;
            cizgi_y = bbox(2) + bbox(4) + 20; 
            plot([bbox(1), bbox(1)+bbox(3)], [cizgi_y, cizgi_y], 'c|-', 'LineWidth', 2, 'MarkerSize', 8);
            text(bbox(1) + bbox(3)/2, cizgi_y + 8, sprintf('%.1f mm', hasar_boyu_mm), ...
                'Color', 'cyan', 'FontSize', 12, 'FontWeight', 'bold', 'HorizontalAlignment', 'center', 'VerticalAlignment', 'top');
        else 
            hasar_boyu_mm = bbox(4) * px_mm;
            cizgi_x = bbox(1) + bbox(3) + 20; 
            plot([cizgi_x, cizgi_x], [bbox(2), bbox(2)+bbox(4)], 'c|-', 'LineWidth', 2, 'MarkerSize', 8);
            text(cizgi_x + 8, bbox(2) + bbox(4)/2, sprintf('%.1f mm', hasar_boyu_mm), ...
                'Color', 'cyan', 'FontSize', 12, 'FontWeight', 'bold', 'HorizontalAlignment', 'left', 'VerticalAlignment', 'middle');
        end
    end
end
info_txt = sprintf('YOLO: %s (%%%.1f)  |  Çap: %.2f mm', strrep(upper(yolo_sinif), '_', ' '), yolo_guven, cap_mm);
text(W/2, H - 15, info_txt, 'Color', yolo_risk_renk, 'FontSize', 14, 'FontWeight', 'bold', 'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom', 'BackgroundColor', [0 0 0 0.8]);
title(ax_detay, sprintf('DETAYLI ANALİZ | Risk: %s', yolo_risk_str), 'Color', yolo_risk_renk, 'FontSize', 14, 'FontWeight', 'bold');
set(ax_detay, 'Color',[0.07 0.07 0.10], 'XColor','none', 'YColor','none');
hold off;
for ax = [ax1 ax2 ax3 ax4 ax5 ax6]
    set(ax, 'Color',[0.07 0.07 0.10], 'XColor','none', 'YColor','none');
end
fprintf('[✓] Analiz başarıyla tamamlandı.\n');

%% ─── 8. VERİ KAYDI (HUFFMAN MAT, TXT RAPOR VE EXCEL/CSV LOG) ────────────
fprintf('\n╔─ DOSYA KAYDI VE SIKIŞTIRMA (HUFFMAN) ──╗\n');
fprintf('║ Veriler "sonuc" klasörüne işleniyor...                                    ║\n');
cikis_klasoru = 'sonuc';
if ~exist(cikis_klasoru, 'dir')
    mkdir(cikis_klasoru);
end
[~, ham_isim, ~] = fileparts(dosya);
zaman_damgasi = datestr(now, 'yyyymmdd_HHMMSS'); 
isim = sprintf('%s_%s', ham_isim, zaman_damgasi); 
rapor_adi = fullfile(cikis_klasoru, sprintf('%s_analiz_raporu.txt', isim));
fid = fopen(rapor_adi, 'w', 'n', 'utf-8');
fprintf(fid, 'KABLO ANALİZ RAPORU\n====================================\n');
fprintf(fid, 'Tarih         : %s\nDosya Adı     : %s\n', datestr(now, 'dd-mm-yyyy HH:MM:SS'), dosya);
fprintf(fid, 'Kablo Çapı    : %.2f mm (%.1f px)\nYOLO Teşhisi  : %s (Güven: %%%.1f)\n', cap_mm, cap_px, yolo_sinif, yolo_guven);
fprintf(fid, 'Genel Risk    : %s\n------------------------------------\n', yolo_risk_str);
if ~strcmp(yolo_sinif, 'SAĞLAM')
    fprintf(fid, 'HASAR DETAYLARI (YOLO):\n');
    for b = 1:num_yolo_boxes
        bbox = yolo_bboxes(b, :);
        if abs(aci) <= 45, has_boyu = bbox(3) * px_mm; else, has_boyu = bbox(4) * px_mm; end
        fprintf(fid, '- Hasar %d Uzunluğu : %.1f mm\n', b, has_boyu);
    end
end
fprintf(fid, '====================================\n');
fclose(fid);
if isfile('yolo_sonuc.txt')
    movefile('yolo_sonuc.txt', fullfile(cikis_klasoru, sprintf('%s_yolo_ham.txt', isim)));
end
Tarih_Saat = string(datestr(now, 'yyyy-mm-dd HH:MM:SS'));
Dosya_Adi  = string(dosya);
Kablo_Capi = round(cap_mm, 2);
YOLO_Sonuc = string(yolo_sinif);
Guven_Skoru = round(yolo_guven, 1);
Hasar_Adedi = num_yolo_boxes; 
Genel_Durum = string(yolo_risk_str);
yeni_kayit = table(Tarih_Saat, Dosya_Adi, Kablo_Capi, YOLO_Sonuc, Guven_Skoru, Hasar_Adedi, Genel_Durum, ...
    'VariableNames', {'Tarih', 'Dosya_Adi', 'Kablo_Capi_mm', 'YOLO_Teshisi', 'Guven_Skoru', 'Hasar_Sayisi', 'Genel_Risk'});
excel_yolu = fullfile(cikis_klasoru, 'TUM_ANALIZ_KAYITLARI.xlsx');
csv_yolu   = fullfile(cikis_klasoru, 'TUM_ANALIZ_KAYITLARI.csv');
try
    if isfile(excel_yolu)
        writetable(yeni_kayit, excel_yolu, 'WriteMode', 'Append');
        writetable(yeni_kayit, csv_yolu, 'WriteMode', 'Append');
    else
        writetable(yeni_kayit, excel_yolu);
        writetable(yeni_kayit, csv_yolu);
    end
    csv_basarili = true;
catch
    csv_basarili = false;
end
analiz_verisi = struct();
analiz_verisi.Dosya_Adi = dosya;
analiz_verisi.Kablo_Capi_mm = cap_mm;
analiz_verisi.Yolo_Teshisi = yolo_risk_str;
analiz_verisi.Hasar_Kutulari = yolo_bboxes;
if size(orijinal, 3) == 3
    gri_kayit = rgb2gray(orijinal);
else
    gri_kayit = orijinal; 
end
[satir_sayisi, sutun_sayisi] = size(gri_kayit);
vektor_img = double(gri_kayit(:));
[semboller, ~, idx] = unique(vektor_img);
frekanslar = accumarray(idx, 1);
olasiliklar = frekanslar / length(vektor_img);
try
    [sozluk, ~] = huffmandict(semboller, olasiliklar);
    sikistirilmis_kod = huffmanenco(vektor_img, sozluk);
    orijinal_boyut_bit = satir_sayisi * sutun_sayisi * 8; 
    yeni_boyut_bit = length(sikistirilmis_kod);
    kazanc_orani = (1 - (yeni_boyut_bit / orijinal_boyut_bit)) * 100;
    mat_kayit_adi = fullfile(cikis_klasoru, sprintf('%s_huffman_verisi.mat', isim));
    save(mat_kayit_adi, 'analiz_verisi', 'sikistirilmis_kod', 'sozluk', 'satir_sayisi', 'sutun_sayisi');
    fprintf('║ [✓] İşlem Başarılı!                                                                                \n');
    fprintf('║ Klasör           : /%s                                                                              \n', cikis_klasoru);
    fprintf('║ Huffman Sıkış.   : %%%-5.2f                                                               \n', kazanc_orani);
catch ME
    fprintf('║ [!] Huffman Hatalı (Toolbox eksik)                                                       \n');
    fprintf('║ TXT, CSV ve Excel raporları kaydedildi.                                             \n');
end
if ~csv_basarili, fprintf('║ [X] UYARI: Excel/CSV açık olabilir!                               \n'); end
fprintf('╚─────────────────────────────────────────╝\n');