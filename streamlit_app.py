import streamlit as st
import pandas as pd
import joblib

print("🚀 Streamlit sta eseguendo il codice...")

@st.cache_resource()
def load_model():
    try:
        model = joblib.load("modello_finale_v1.3.joblib")
        st.success("✅ Modello caricato correttamente.")
        return model
    except Exception as e:
        st.error(f"❌ Errore nel caricamento modello: {e}")
        print(f"❌ Errore nel caricamento modello: {e}")
        return None

model = load_model()
if model is None:
    st.error("Impossibile procedere senza un modello valido.")
    st.stop()

st.title("Predizione del Rischio Cardiovascolare - Modello Multimodale")
st.sidebar.header("Inserisci i tuoi dati clinici e genetici")

age = st.sidebar.slider("Età", 40, 90, 65)
bmi = st.sidebar.number_input("BMI", value=27.0)
crp = st.sidebar.number_input("CRP (mg/L)", value=2.5)
cholesterol = st.sidebar.number_input("Colesterolo (mg/dL)", value=200.0)
nlr = st.sidebar.number_input("NLR", value=2.5)
plr = st.sidebar.number_input("PLR", value=130.0)

st.sidebar.markdown("---")
st.sidebar.markdown("### Mutazioni CHIP")
dnmt3a = st.sidebar.checkbox("DNMT3A")
tet2 = st.sidebar.checkbox("TET2")
asxl1 = st.sidebar.checkbox("ASXL1")

dnmt3a_vaf = st.sidebar.slider("DNMT3A VAF", 0.0, 0.4, 0.0) if dnmt3a else 0.0
tet2_vaf = st.sidebar.slider("TET2 VAF", 0.0, 0.4, 0.0) if tet2 else 0.0
asxl1_vaf = st.sidebar.slider("ASXL1 VAF", 0.0, 0.4, 0.0) if asxl1 else 0.0

chip_burden = dnmt3a_vaf + tet2_vaf + asxl1_vaf

input_data = pd.DataFrame([{
    "age": age,
    "crp": crp,
    "bmi": bmi,
    "cholesterol": cholesterol,
    "nlr": nlr,
    "plr": plr,
    "DNMT3A": int(dnmt3a),
    "TET2": int(tet2),
    "ASXL1": int(asxl1),
    "DNMT3A_vaf": dnmt3a_vaf,
    "TET2_vaf": tet2_vaf,
    "ASXL1_vaf": asxl1_vaf,
    "chip_burden": chip_burden
}])

st.subheader("Anteprima dati inseriti:")
st.write(input_data)

try:
    risk_proba = model.predict_proba(input_data)[0, 1]
    st.subheader("Risultato della predizione")
    st.metric("Rischio stimato di evento cardiovascolare (%)", f"{risk_proba*100:.1f}%")

    if chip_burden > 0.2:
        st.warning("Alto CHIP burden rilevato. Potenziale rischio aggiuntivo mediato da infiammazione clonale.")
    elif risk_proba > 0.3:
        st.info("Il rischio stimato è moderato-alto. Si consiglia valutazione clinica specialistica.")
    else:
        st.success("Rischio stimato contenuto. Mantenere monitoraggio periodico.")
except Exception as e:
    st.error(f"❌ Errore nella predizione: {e}")

st.markdown("---")
st.subheader("Dati inseriti")
st.dataframe(input_data.T, use_container_width=True)
