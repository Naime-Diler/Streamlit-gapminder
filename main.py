import streamlit as st
import plotly.express as px
import joblib
import pandas as pd

st.set_page_config(layout="wide")


@st.cache_data
def get_data():
    df = px.data.gapminder()
    return df

def get_model():
    model = joblib.load("gapminder_model.joblib")
    return model

st.header("👩‍⚕️:red[Prognose] der :red[Lebenserwartung]👨‍⚕️")

tab_home, tab_vis, tab_model = st.tabs(["Hauptseite", "Grafiken", "Modell"])

# Tab Home
col_left, col_right = tab_home.columns(2, gap="large")

col_left.subheader("Wer war Hans Rosling?")

col_left.markdown("**Hans Rosling** wurde 1948 in Uppsala, Schweden, geboren. Er war nicht nur Arzt und Professor für internationale Gesundheit, sondern auch ein charismatischer Redner, der mit seinen öffentlichen Vorträgen Millionen von Menschen begeisterte. Als Berater für die Weltgesundheitsorganisation (WHO) und UNICEF setzte er sich für globale Gesundheitsfragen ein. Zudem war er Mitbegründer von Ärzte ohne Grenzen in der Schweiz sowie der Gapminder-Stiftung, die sich der Vermittlung von Daten und Fakten über die Welt verschrieben hat. Seine TED-Talks wurden über 35 Millionen Mal angesehen, und das *Time Magazine* zählte ihn zu den 100 einflussreichsten Persönlichkeiten der Welt. Die letzten zehn Jahre seines Lebens widmete er dem Schreiben seines Buches *Factfulness*, das noch heute große Bedeutung hat. Hans Rosling verstarb 2017, doch sein Erbe lebt weiter und inspiriert Menschen weltweit.")

col_left.image("hans_rosling.png")
col.left.markdown("https://media.eagereyes.org/wp-content/uploads/2017/02/hans-rosling-tc14.jpg")


col_right.subheader("Der Datensatz")
col_right.markdown("Frühere Studien haben sich bereits mit dem Einfluss demografischer Faktoren, Einkommensverteilung und Sterberaten befasst. Der Einfluss von Impfungen und des Human Development Index (HDI) wurde jedoch oft außer Acht gelassen. Diese Untersuchung soll daher nicht nur wirtschaftliche und soziale Faktoren, sondern auch Immunisierungs- und Sterblichkeitsraten sowie weitere gesundheitsrelevante Aspekte berücksichtigen. Da die Daten länderbasiert sind, lassen sich die Schlüsselfaktoren, die die Lebenserwartung einer Bevölkerung beeinflussen, leichter identifizieren. Dies ermöglicht es Ländern, gezielt Maßnahmen zu ergreifen, um die Lebensqualität und -erwartung ihrer Bevölkerung nachhaltig zu verbessern.")

df = get_data()
col_right.dataframe(df)

# Tab Vis
# Grafik 1

tab_vis.subheader("Vergleich der Lebenserwartung ausgewählter Länder im Zeitverlauf")

selected_countries = tab_vis.multiselect(label="Wählen Sie ein Land aus", options=df.country.unique(), default=["Turkey", "Syria", "Greece"])

# tab_vis.write(selected_countries)

filtered_df = df[df.country.isin(selected_countries)]

fig = px.line(
    filtered_df,
    x="year",
    y="lifeExp",
    color="country"
)

tab_vis.plotly_chart(fig, use_container_width=True)

# Grafik 2

tab_vis.subheader("Visualisierung der Veränderung der Lebenserwartung in den Ländern über die Jahre auf einer Karte.")

year_select_for_map = tab_vis.slider("Jahre", min_value=int(df.year.min()), max_value=int(df.year.max()),
                                     step=5)

fig2 = px.choropleth(df[df.year == year_select_for_map], locations="iso_alpha",
                     color="lifeExp",
                     range_color=(df.lifeExp.min(), df.lifeExp.max()),
                     hover_name="country",
                     color_continuous_scale=px.colors.sequential.Plasma)

tab_vis.plotly_chart(fig2, use_container_width=True)

# Grafik 3
tab_vis.subheader("Entwicklung von Bevölkerung, BIP und Lebenserwartung der Länder über die Jahre.")

fig3 = px.scatter(df, x="gdpPercap", y="lifeExp", size="pop", color="continent",
                  animation_group="country", animation_frame="year",
                  hover_name="country", range_x=[100, 100000], range_y=[25, 90], log_x=True, size_max=60)

fig3.add_hline(y=50, line_dash="dash", line_color="white")
tab_vis.plotly_chart(fig3, use_container_width=True)

# Tab Model:

model = get_model()

year = tab_model.number_input("Jahr", min_value=1952, max_value=2027, step=1, value=2000)
pop = tab_model.number_input("Bevölkerung", min_value=10000, max_value=1000000000, step=100000, value=1000000)
gdpPercap = tab_model.number_input("BIP", min_value=1, step=1, value=5000)

user_input = pd.DataFrame({"year":year, "pop":pop, "gdpPercap":gdpPercap}, index=[0])

# tab_model.write(user_input)

if tab_model.button("Vorhersage starten"):
    prediction = model.predict(user_input)
    tab_model.success(f"Prognostizierte Lebenserwartung: {prediction}")
    st.balloons()
