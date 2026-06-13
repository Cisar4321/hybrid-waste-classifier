import sys
from pathlib import Path

import numpy as np
import cv2
import joblib
import streamlit as st
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops
from skimage.filters import gabor_kernel
from skimage.measure import shannon_entropy
from scipy.stats import skew, kurtosis

ROOT = Path(__file__).parent
sys.path.append(str(ROOT))
from src.segmentation.edge_detection import bordes_sv, densidad_bordes
from src.segmentation.morphological_ops import rellenar_contornos

MODEL_DIR = ROOT / 'results' / 'models'
UMBRAL_DENSIDAD = 0.15
K_GAUSS = 11
IMG_SIZE = (224, 224)

LBP_P, LBP_R = 8, 1
LBP_BINS = LBP_P + 2
GLCM_DIST = [1]
GLCM_ANG = [0, np.pi/4, np.pi/2, 3*np.pi/4]
GLCM_PROPS = ['contrast', 'homogeneity', 'energy', 'correlation']
GABOR_FREQS = [0.1, 0.2, 0.3]
GABOR_THETAS = [0, np.pi/4, np.pi/2, 3*np.pi/4]
GABOR_KERNELS = [(f, t, np.real(gabor_kernel(f, theta=t)))
                 for f in GABOR_FREQS for t in GABOR_THETAS]

TACHOS = {
    'battery':   ('Residuos peligrosos', 'Rojo',    '#c0392b', '#ffffff', 'Pilas y baterias a punto de acopio especial, nunca al tacho comun.'),
    'cardboard': ('Papel y carton',      'Azul',    '#2563eb', '#ffffff', 'Aplasta las cajas y mantenlas secas.'),
    'paper':     ('Papel y carton',      'Azul',    '#2563eb', '#ffffff', 'Sin restos de comida ni grasa.'),
    'glass':     ('Vidrio',              'Verde',   '#16a34a', '#ffffff', 'Enjuaga y retira tapas.'),
    'metal':     ('Metales',             'Amarillo','#eab308', '#1a1207', 'Enjuaga las latas antes de desechar.'),
    'plastic':   ('Plasticos',           'Blanco',  '#cbd5e1', '#0a0e12', 'Enjuaga botellas y envases.'),
    'organic':   ('Organicos',           'Marron',  '#92400e', '#ffffff', 'Restos de comida y poda.'),
    'textile':   ('No aprovechables',    'Negro',   '#1f2933', '#ffffff', 'Si esta en buen estado, considera donarlo.'),
    'trash':     ('No aprovechables',    'Negro',   '#1f2933', '#ffffff', 'Residuo general no reciclable.'),
}


@st.cache_resource
def cargar():
    svm = joblib.load(MODEL_DIR / 'svm_clasico.pkl')
    scaler = joblib.load(MODEL_DIR / 'scaler_clasico.pkl')
    features = joblib.load(MODEL_DIR / 'features_clasico.pkl')
    from tensorflow.keras.models import load_model
    cnn = load_model(MODEL_DIR / 'cnn_verificador.h5')
    return svm, scaler, features, cnn


def segmentar(img_bgr):
    hsv_raw = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    S_raw = cv2.GaussianBlur(hsv_raw[:, :, 1], (5, 5), 0)
    densidad = densidad_bordes(S_raw, K_GAUSS)
    aplico_lp = densidad > UMBRAL_DENSIDAD
    img_lp = cv2.bilateralFilter(img_bgr, 50, 220, 300) if aplico_lp else img_bgr.copy()
    img_hsv = cv2.cvtColor(img_lp, cv2.COLOR_BGR2HSV)
    _, S, V = cv2.split(img_hsv)
    sobel_comb = bordes_sv(S, V, K_GAUSS)
    _, cerrado, relleno, k_open, k_close = rellenar_contornos(sobel_comb)
    pasos = {
        'S': S, 'sobel': sobel_comb, 'cerrado': cerrado, 'relleno': relleno,
        'densidad': densidad, 'aplico_lp': aplico_lp, 'k_open': k_open, 'k_close': k_close,
    }
    return relleno, pasos


def f_textura(gray, mask=None):
    f = {}
    if mask is not None:
        ys, xs = np.where(mask > 0)
        if len(xs) == 0:
            f.update({f'lbp_{i}': 0 for i in range(LBP_BINS)})
            f.update({f'glcm_{p}': 0 for p in GLCM_PROPS})
            return f
        region = gray[ys.min():ys.max()+1, xs.min():xs.max()+1]
    else:
        region = gray
    lbp = local_binary_pattern(region, LBP_P, LBP_R, method='uniform')
    hist, _ = np.histogram(lbp.ravel(), bins=LBP_BINS, range=(0, LBP_BINS), density=True)
    for i, h in enumerate(hist):
        f[f'lbp_{i}'] = h
    glcm = graycomatrix(region, GLCM_DIST, GLCM_ANG, levels=256, symmetric=True, normed=True)
    for prop in GLCM_PROPS:
        f[f'glcm_{prop}'] = graycoprops(glcm, prop).mean()
    return f


def f_gabor(gray, mask=None):
    f = {}
    if mask is not None:
        ys, xs = np.where(mask > 0)
        if len(xs) == 0:
            for i in range(len(GABOR_KERNELS)):
                f[f'gabor_{i}_mean'] = 0
                f[f'gabor_{i}_std'] = 0
            return f
        region = gray[ys.min():ys.max()+1, xs.min():xs.max()+1].astype(np.float64)
    else:
        region = gray.astype(np.float64)
    for i, (_, _, kernel) in enumerate(GABOR_KERNELS):
        filt = cv2.filter2D(region, cv2.CV_64F, kernel)
        f[f'gabor_{i}_mean'] = filt.mean()
        f[f'gabor_{i}_std'] = filt.std()
    return f


def f_intensidad(gray, mask=None):
    vals = gray[mask > 0].astype(np.float64) if mask is not None else gray.ravel().astype(np.float64)
    if vals.size == 0:
        return {k: 0 for k in ['int_mean', 'int_std', 'int_skew', 'int_kurt', 'int_entropia']}
    return {'int_mean': vals.mean(), 'int_std': vals.std(), 'int_skew': skew(vals),
            'int_kurt': kurtosis(vals), 'int_entropia': shannon_entropy(vals)}


def f_forma(mask, h, w):
    contornos, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contornos:
        return {k: 0 for k in ['area', 'perimetro', 'extent', 'solidity', 'aspect_ratio', 'circularidad']}
    c = max(contornos, key=cv2.contourArea)
    area = cv2.contourArea(c)
    perim = cv2.arcLength(c, True)
    x, y, bw, bh = cv2.boundingRect(c)
    area_hull = cv2.contourArea(cv2.convexHull(c))
    return {
        'area': area / (h * w),
        'perimetro': perim / (2 * (h + w)),
        'extent': area / (bw * bh) if bw * bh > 0 else 0,
        'solidity': area / area_hull if area_hull > 0 else 0,
        'aspect_ratio': bw / bh if bh > 0 else 0,
        'circularidad': 4 * np.pi * area / (perim ** 2) if perim > 0 else 0,
    }


def f_hu(mask):
    hu = cv2.HuMoments(cv2.moments(mask)).flatten()
    return {f'hu_{i}': -np.sign(h) * np.log10(abs(h) + 1e-12) for i, h in enumerate(hu)}


def f_color(hsv, mask):
    f = {}
    for i, ch in enumerate(['h', 's', 'v']):
        vals = hsv[:, :, i][mask > 0]
        f[f'color_{ch}_mean'] = vals.mean() if vals.size else 0
        f[f'color_{ch}_std'] = vals.std() if vals.size else 0
    return f


def f_color_extra(img_bgr, mask):
    f = {}
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    for i, ch in enumerate(['h', 's', 'v']):
        vals = hsv[:, :, i][mask > 0]
        f[f'color_{ch}_med'] = np.median(vals) if vals.size else 0
        f[f'color_{ch}_p25'] = np.percentile(vals, 25) if vals.size else 0
        f[f'color_{ch}_p75'] = np.percentile(vals, 75) if vals.size else 0
    for i, ch in zip([1, 2], ['a', 'b']):
        vals = lab[:, :, i][mask > 0]
        f[f'lab_{ch}_mean'] = vals.mean() if vals.size else 0
        f[f'lab_{ch}_std'] = vals.std() if vals.size else 0
    return f


def vector_features(img_bgr, relleno, columnas):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    fila = {}
    for k, v in f_textura(gray, relleno).items():
        fila[f'obj_{k}'] = v
    for k, v in f_textura(gray, None).items():
        fila[f'full_{k}'] = v
    for k, v in f_gabor(gray, relleno).items():
        fila[f'obj_{k}'] = v
    for k, v in f_gabor(gray, None).items():
        fila[f'full_{k}'] = v
    for k, v in f_intensidad(gray, relleno).items():
        fila[k] = v
    for k, v in f_forma(relleno, h, w).items():
        fila[f'forma_{k}'] = v
    for k, v in f_hu(relleno).items():
        fila[f'forma_{k}'] = v
    for k, v in f_color(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV), relleno).items():
        fila[k] = v
    for k, v in f_color_extra(img_bgr, relleno).items():
        fila[k] = v
    vec = np.array([fila.get(c, 0) for c in columnas], dtype=np.float64)
    vec[~np.isfinite(vec)] = 0
    if 'forma_aspect_ratio' in columnas:
        idx = columnas.index('forma_aspect_ratio')
        vec[idx] = min(vec[idx], 5)
    return vec


def proba_svm(svm, scaler, features, img_bgr, relleno):
    vec = vector_features(img_bgr, relleno, features)
    Xs = scaler.transform(vec.reshape(1, -1))
    return svm.predict_proba(Xs)[0]


def proba_cnn(cnn, img_bgr):
    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, IMG_SIZE).astype('float32') / 255.0
    return cnn.predict(np.expand_dims(img, 0), verbose=0)[0]


st.set_page_config(page_title='Clasificador de residuos', layout='wide')
st.markdown("""
<style>
.block-container{padding-top:2rem}
.verdict{border-radius:14px;padding:22px;margin-bottom:8px}
.verdict h2{margin:0;font-size:30px}
.verdict p{margin:4px 0 0;font-family:monospace;opacity:.9}
.chip{display:inline-block;padding:8px 12px;border-radius:8px;margin:3px;font-size:13px;font-family:monospace}
</style>""", unsafe_allow_html=True)

st.title('Clasificador hibrido de residuos')
st.caption('SVM clasico (LBP, GLCM, Gabor, HSV/LAB, forma) con verificador CNN MobileNetV2')

try:
    svm, scaler, features, cnn = cargar()
    clases_svm = list(svm.classes_)
    clases_cnn = sorted(TACHOS.keys())
except Exception as e:
    st.error(f'No se cargaron los modelos en results/models. Detalle: {e}')
    st.stop()

st.header('El problema que resolvemos')
e1, e2 = st.columns(2)
with e1:
    st.markdown("""
Mucho reciclaje se pierde por **mala segregacion en el punto de origen**: un solo residuo
mal puesto (vidrio, una pila) puede contaminar todo un lote de reciclables, y la gente a
menudo no sabe en que tacho va cada objeto.

Esta herramienta asiste esa decision: subes una foto y sugiere el contenedor correcto,
con el porque.

**Limites a tener en cuenta.** Varias clases se confunden por color o textura (vidrio, pila,
papel, plastico con fondos claros) y el set de entrenamiento esta balanceado artificialmente.
Por eso conviene usarlo como **asistente** que reduce el error humano, no como decisor autonomo.
""")
with e2:
    st.markdown("""
**Por que un pipeline hibrido.** El SVM clasico es rapido y barato: resuelve los casos
faciles con descriptores hechos a mano (textura, color, forma). Cuando duda (confianza baja),
recien entra la CNN, mas precisa pero mas costosa. Asi se ahorra computo, lo que importa
para correr en dispositivos de bajo recurso (edge).

**Las etapas del flujo:**
1. Segmentacion: aisla el objeto del fondo (saturacion, bordes Sobel, morfologia).
2. Extraccion de features sobre el objeto.
3. Clasificacion (SVM, CNN o hibrido).
4. Mapeo de la clase al tacho correspondiente.
""")

st.markdown('**Guia de tachos (las 9 clases):**')
chips = ''.join(
    f"<span class='chip' style='background:{v[2]};color:{v[3]}'>{k} → {v[0]} ({v[1]})</span>"
    for k, v in TACHOS.items())
st.markdown(chips, unsafe_allow_html=True)
st.divider()

with st.sidebar:
    st.header('Configuracion')
    modo = st.radio('Modo de clasificacion', ['SVM solo', 'Hibrido SVM + CNN', 'CNN sola'])
    theta = st.slider('Umbral de confianza (theta)', 0.30, 0.95, 0.60, 0.05,
                      disabled=(modo != 'Hibrido SVM + CNN'))
    ver_pre = st.checkbox('Mostrar preprocesamiento', value=True)
    st.caption('En modo hibrido, si la confianza del SVM < theta la imagen pasa al verificador CNN.')

archivos = st.file_uploader('Sube una o varias imagenes', type=['jpg', 'jpeg', 'png'],
                            accept_multiple_files=True)

for archivo in archivos or []:
    st.divider()
    st.subheader(archivo.name)

    data = np.frombuffer(archivo.read(), np.uint8)
    img_bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    with st.spinner('Procesando...'):
        relleno, pasos = segmentar(img_bgr)

        ps = proba_svm(svm, scaler, features, img_bgr, relleno) if modo != 'CNN sola' else None
        pc = proba_cnn(cnn, img_bgr) if modo != 'SVM solo' else None

        if modo == 'SVM solo':
            clases, proba = clases_svm, ps
            clase = clases[int(np.argmax(proba))]
            conf, fuente = float(proba.max()), 'SVM'
        elif modo == 'CNN sola':
            clases, proba = clases_cnn, pc
            clase = clases[int(np.argmax(proba))]
            conf, fuente = float(proba.max()), 'CNN'
        else:
            conf_svm = float(ps.max())
            if conf_svm >= theta:
                clases, proba = clases_svm, ps
                clase, conf, fuente = clases[int(np.argmax(ps))], conf_svm, 'SVM'
            else:
                clases, proba = clases_cnn, pc
                clase, conf, fuente = clases[int(np.argmax(pc))], float(pc.max()), 'CNN'

    tacho, cont, hex_, fg, tip = TACHOS[clase]
    c1, c2 = st.columns([1, 1])
    with c1:
        st.image(img_rgb, caption='Entrada', use_container_width=True)
    with c2:
        st.markdown(
            f"<div class='verdict' style='background:{hex_};color:{fg}'>"
            f"<p>Disposicion sugerida · clase {clase}</p>"
            f"<h2>{tacho}</h2><p>Contenedor · {cont}</p></div>",
            unsafe_allow_html=True)
        st.info(tip)
        a, b, d = st.columns(3)
        a.metric('Confianza', f'{conf:.1%}')
        b.metric('Resuelto por', fuente)
        d.metric('Modo', modo.split()[0])
        st.markdown('**Distribucion de probabilidad**')
        st.bar_chart({c: float(p) for c, p in zip(clases, proba)})

    if ver_pre:
        with st.expander('Proceso de preprocesamiento y segmentacion', expanded=False):
            m1, m2, m3, m4 = st.columns(4)
            m1.metric('Densidad S', f"{pasos['densidad']:.3f}")
            m2.metric('Filtro pasabajo', 'Si' if pasos['aplico_lp'] else 'No')
            m3.metric('k_open / k_close', f"{pasos['k_open']} / {pasos['k_close']}")
            cobertura = 100 * np.count_nonzero(pasos['relleno']) / pasos['relleno'].size
            m4.metric('Cobertura objeto', f'{cobertura:.1f}%')

            objeto = cv2.bitwise_and(img_rgb, img_rgb, mask=pasos['relleno'])
            overlay = img_rgb.copy()
            overlay[pasos['relleno'] > 0] = (0.5 * overlay[pasos['relleno'] > 0]
                                             + 0.5 * np.array([255, 0, 0])).astype(np.uint8)

            p1, p2, p3 = st.columns(3)
            p1.image(pasos['S'], caption='Canal S (saturacion)', use_container_width=True, clamp=True)
            p2.image(pasos['sobel'], caption='Bordes Sobel S|V', use_container_width=True, clamp=True)
            p3.image(pasos['cerrado'], caption='Morfologia (closing)', use_container_width=True, clamp=True)
            p4, p5, p6 = st.columns(3)
            p4.image(pasos['relleno'], caption='Mascara final', use_container_width=True, clamp=True)
            p5.image(objeto, caption='Objeto segmentado', use_container_width=True)
            p6.image(overlay, caption='Overlay', use_container_width=True)