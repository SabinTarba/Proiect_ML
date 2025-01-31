import streamlit as st
import pandas as pd
import time
import numpy as np
import random
from sqlalchemy import create_engine
from io import BytesIO
import plotly.express as px
import seaborn
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import Normalizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

@st.cache_data 
def load_data():
    username = "sabintarba"      
    password = "sabintarba"      
    host = "localhost"              
    port = "1521"              
    service_name = "ORCL"   

    oracle_connection_string = f'oracle+cx_oracle://{username}:{password}@{host}:{port}/?service_name={service_name}'
    engine = create_engine(oracle_connection_string)

    sql_query = "SELECT * FROM movies"

    df = pd.read_sql(sql_query, engine)

    df["genres"] = df['genres'].str.split(",")
    df["actors"] = df['actors'].str.split(",")
    
    return df

def page_intro():
    st.markdown("""
            ### Obtinerea datelor
            Datele au fost preluate de pe kaggle.com (https://www.kaggle.com/datasets/yusufdelikkaya/imdb-movie-dataset).

            ### Descrierea datelor
            - Setul de date cuprinde date anonimizate despre filmele disponibile pe IMDb, surprinzând diverse aspecte, cum ar fi genul, ratingul și veniturile.
            - Acest set de date poate fi utilizat pentru a analiza tendințele filmelor, preferințele publicului și impactul diferitelor atribute, cum ar fi genul și regizorul, asupra succesului filmului.
            - Poate ajuta la înțelegerea factorilor care contribuie la ratingurile ridicate și la veniturile de box office, precum și la furnizarea de informații despre popularitatea genurilor de-a lungul timpului.
            - Acest set de date poate fi utilizat pentru a analiza factorii de succes a filmului, preferințele publicului și tendințele genurilor.
            - Poate ajuta la identificarea relației dintre caracteristicile filmului (de exemplu, gen, regizor) și ratinguri sau venituri, la examinarea popularității actorilor și regizorilor și la înțelegerea recepției critice prin Metascore.
            
            ### Scop
            Scopul aplicatiei este de a oferi o perspectiva asupra filmelor inregistrate si de a face o analiza descriptiva a acestora.
        """)

def page_data_import():
    st.markdown("""
            ### Importul datelor
            - Datele au fost descarcate de la adresa mentionata in pagina "Introducere" sub format CSV. 
            - Randurile au urmatoarea strucutra: \n
            ```csv
            Rank,Title,Genre,Description,Director,Actors,Year,Runtime (Minutes),Rating,Votes,Revenue (Millions),Metascor
            1,Guardians of the Galaxy,"Action,Adventure,Sci-Fi",A group of intergalactic criminals are forced to work together to stop a fanatical warrior from taking control of the universe.,James Gunn,"Chris Pratt, Vin Diesel, Bradley Cooper, Zoe Saldana",2014,121,8.1,757074,333.13,76
            2,Prometheus,"Adventure,Mystery,Sci-Fi","Following clues to the origin of mankind, a team finds a structure on a distant moon, but they soon realize they are not alone.",Ridley Scott,"Noomi Rapace, Logan Marshall-Green, Michael Fassbender, Charlize Theron",2012,124,7,485820,126.46,65
            3,Split,"Horror,Thriller",Three girls are kidnapped by a man with a diagnosed 23 distinct personalities. They must try to escape before the apparent emergence of a frightful new 24th.,M. Night Shyamalan,"James McAvoy, Anya Taylor-Joy, Haley Lu Richardson, Jessica Sula",2016,117,7.3,157606,138.12,62
            4,Sing,"Animation,Comedy,Family","In a city of humanoid animals, a hustling theater impresario's attempt to save his theater with a singing competition becomes grander than he anticipates even as its finalists' find that their lives will never be the same.",Christophe Lourdelet,"Matthew McConaughey,Reese Witherspoon, Seth MacFarlane, Scarlett Johansson",2016,108,7.2,60545,270.32,59
            5,Suicide Squad,"Action,Adventure,Fantasy",A secret government agency recruits some of the most dangerous incarcerated super-villains to form a defensive task force. Their first mission: save the world from the apocalypse.,David Ayer,"Will Smith, Jared Leto, Margot Robbie, Viola Davis",2016,123,6.2,393727,325.02,40
            ```    
            - Acestea au fost incarcate ulterior intr-o baza de date Oracle utilizand o tabela externa de forma: \n
            ```plsql
            create or replace directory DIR as 'C:\\Users\\Sabin Tarba\\Desktop\\Proiect_AVMD';
                
            create table movies$ext (
                id               number,
                title            varchar2(100),
                genres           varchar2(100),
                description      varchar2(1000),
                director         varchar2(100),
                actors           varchar2(500),
                release_year     number(19),
                duration         number(19),
                rating           number(19),
                votes            number(19),
                revenue_millions number(19),
                metascore        number(19)
            )
            organization external
            (
                type oracle_loader
                default directory dir
                access parameters
                (
                    records delimited by newline
                    skip 1
                    fields terminated by ',' optionally enclosed by '"'
                    missing field values are null
                    (
                        id               char(10),
                        title            char(100),
                        genres           char(100),
                        description      char(1000),
                        director         char(100),
                        actors           char(500),
                        release_year     char(10),
                        duration         char(10),
                        rating           char(10),
                        votes            char(10),
                        revenue_millions char(10),
                        metascore        char(10)
                    )
                )
                location ('movies.csv')
            )
            reject limit unlimited;
            ```
            - In final datele din tabela externa au fost incarcate intr-o tabela normala cu strucutra identica prin: \n
            ```plsql
            create table movies as select * from movies$ext;
            ```
        """)

def page_get_data_from_db():

    st.markdown("""
    ### Incarcarea datelor
    Incarcarea datelor se realizeaza prin deschiderea unei conexiuni catre baza de date, preluarea datelor intr-un cursor si apoi incarcarea acestuia intr-un obiect de tip DataFrame.
    """)

    clicked = st.button("Incarca datele", type="primary", use_container_width=True)

    if clicked:
        my_bar = st.progress(0, text="Preluare date din Oracle si incarcare in DataFrame ...")

        bar_completed = False

        percent_complete = 0

        while not bar_completed:
            time.sleep(0.1)
            
            percent_complete = percent_complete + random.randint(6, 15) * 2 - 12

            if percent_complete > 100:
                percent_complete = 100
            
            if percent_complete == 100:
                bar_completed = True

            my_bar.progress(percent_complete, text="Preluare date din Oracle si incarcare in DataFrame ...")
        
        time.sleep(0.5)
        my_bar.empty()

        st.success("Datele au fost incarcate cu succes!")

        if 'df' not in st.session_state:
            df = load_data()

            st.subheader("Date initiale")
            st.dataframe(df)

            # Preprocessing data
            df_numeric = df.select_dtypes(include=[np.number])
            numeric_cols = df_numeric.columns.values.tolist()
            st.write("Coloane numerice: " + ", ".join(numeric_cols))

            df_non_numeric = df.select_dtypes(exclude=[np.number])
            non_numeric_cols = df_non_numeric.columns.values
            st.write("Coloane non-numerice: " +  ", ".join(non_numeric_cols))

            cols = df.columns
            colours = ['#000099', '#ffff00']
            
            st.subheader("Heatmap pentru aflarea coloanelor pentru care exista valori lipsa")
            fig, ax = plt.subplots(figsize=(8, 6))
            seaborn.heatmap(df[cols].isnull(), annot=True, cmap=seaborn.color_palette(colours), ax=ax)
            st.pyplot(fig)

            st.subheader("Inlocuire valori lipsa cu media pentru cele numerice si cu modulul pentru cele non-numerice")

            for col in numeric_cols:
                missing = df[col].isnull()
                num_missing = np.sum(missing)

                if num_missing > 0:
                    mean = round(df[col].mean(), 2)
                    df[col] = df[col].fillna(mean)
                    st.write(f"Coloana care contine valori nule: {col}. Valorile nule vor fi inlocuite cu media: {mean}.")

            for col in non_numeric_cols:
                missing = df[col].isnull()
                num_missing = np.sum(missing)

                if num_missing > 0:
                    top = df[col].describe()['top']
                    df[col] = df[col].fillna(top)
                    st.write(f"Coloana care contine valori nule: {col}. Valorile nule vor fi inlocuite cu modulul: {top}.")

            st.subheader("Eliminare duplicate")
            df_duplicates = df['title'].duplicated()
            duplicates_titles = ", ".join(df["title"][df["title"].duplicated()].unique())

            st.write(f"Duplicate gasite in functie de titlu: {df_duplicates.sum()}")
            st.write(f"Titluri duplicate: {duplicates_titles}")

            df.drop_duplicates(subset=["title"], inplace=True)

            # Save in session state
            st.session_state.df = df
        
        st.dataframe(st.session_state.df, use_container_width=True)

    
def page_descriptive_analysis():
    if 'df' not in st.session_state:
        st.error("""Mergi in meniul "Preluare date din Oracle" si incarca datele!""")
    else:
        df = st.session_state.df

        search_title = st.text_input("Cauta dupa titlu", max_chars=100, placeholder="Titlu")

        all_genres = list(pd.Series([genre for sublist in df['genres'] for genre in sublist]).unique())
        all_actors = list(pd.Series([actor for sublist in df['actors'] for actor in sublist]).unique())
        all_directors = list(df["director"].unique())
        genres_list = st.multiselect("Filmul are cel putin un gen din lista", ["Toate"] + all_genres, placeholder="Alege lista de genuri")
        actors_list = st.multiselect("Filmul are cel putin un actor din lista", ["Toti"] + all_actors, placeholder="Alege lista de actori")
        directors_list = st.multiselect("Filmul are cel putin un actor din lista", ["Toti"] + all_directors, placeholder="Alege lista de directori")

        operator = st.selectbox("Anul lansarii filmului", ["=", ">=", "<=", ">", "<"])
        year_option = st.text_input("anul")

        df_filtered = df

        # Search by title
        if search_title:
            df_filtered = df[df["title"].str.contains(search_title, case=False)]
        
        # Check if "Toate" is selected
        if "Toate" in genres_list or not genres_list:
            genres_list = all_genres

        if "Toti" in actors_list or not actors_list:
            actors_list = all_actors

        if "Toti" in directors_list or not directors_list:
            directors_list = all_directors

        # Check by genres list & actors list & directors list
        df_filtered = df_filtered[
            df_filtered["genres"].apply(lambda x: any(genre in x for genre in genres_list)) 
            & df_filtered["actors"].apply(lambda x: any(actor in x for actor in actors_list)) 
            & df_filtered["director"].isin(directors_list)
        ]

        # Check year option
        if year_option:
            if not year_option.isdigit():
                st.error("Scrie un an valid!")
            else:
                match operator:
                    case "=":
                        df_filtered = df_filtered[df_filtered["release_year"] == int(year_option)]
                    case ">=":
                        df_filtered = df_filtered[df_filtered["release_year"] >= int(year_option)]
                    case "<=":
                        df_filtered = df_filtered[df_filtered["release_year"] <= int(year_option)]
                    case ">":
                        df_filtered = df_filtered[df_filtered["release_year"] > int(year_option)]
                    case "<":
                        df_filtered = df_filtered[df_filtered["release_year"] < int(year_option)]


        st.dataframe(df_filtered)

        all_parameters = list(df_filtered.columns)
        parameters_list = st.multiselect("Alege parametrii pentru statisticile descriptive", ['Toti parametrii'] + all_parameters, default=['Toti parametrii'])

        if 'Toti parametrii' in parameters_list:
            parameters_list = all_parameters

        # Statistics
        if not parameters_list:
            st.error("Selecteaza cel putin un parametru pentru a vizualiza statisticile!")
        elif not df_filtered.empty:
            stats_df = pd.DataFrame()
            for param in parameters_list:
                stats = df_filtered[param].describe().rename(param)
                stats_df = pd.concat([stats_df, stats], axis=1)
            st.info("Statistici descriptive pentru selectia facuta:")
            st.dataframe(stats_df)
            
            # Download CSV
            csv = stats_df.to_csv(index=True).encode('utf-8')
            st.download_button(
                label="Descarca datele sub format CSV",
                data=csv,
                file_name='raport_statistici.csv',
                mime='text/csv'
            )

            # Download Excel
            output = BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                stats_df.to_excel(writer, index=True)
            excel_data = output.getvalue()
            st.download_button(
                label="Descarca datele sub format Excel",
                data=excel_data,
                file_name='raport_statistici.xlsx',
                mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
            )

def page_visual_analysis():
    if 'df' not in st.session_state:
        st.error("""Mergi in meniul "Preluare date din Oracle" si incarca datele!""")
    else:
        df = st.session_state.df

        option = st.radio("Selecteaza tipul de analiza sau vizualizare:", ("Analiza standard"))

        if option == "Analiza standard":
            release_year = st.selectbox('Alege anul:', ['Toti anii'] + list(df['release_year'].unique()))
            parameter = st.selectbox('Alege parametrul:', ['duration', 'rating', 'votes', 'metascore', 'revenue_millions'])
            graph_type = st.selectbox("Alege tipul de grafic:", ["Evlolutia incasarilor", "Histograma", "Box Plot"])

            if release_year == 'Toti anii':
                df_release_year = df
            else:
                df_release_year = df[df["release_year"] == release_year]

            if df_release_year.empty or not parameter:
                st.error("Selectati un an un parametru pentru a vizualiza graficul!")
            else:
                df_release_year = df_release_year.sort_values(by="rating")

                if graph_type == "Evlolutia incasarilor":
                    fig = px.line(df_release_year, x='rating', y=parameter, title=f'Evolutia {parameter} raportat la rating')
                    st.plotly_chart(fig, use_container_width=True)
                elif graph_type == "Histograma":
                    fig = px.histogram(df_release_year, x=parameter, nbins=30, title=f'Histograma {parameter}')
                    st.plotly_chart(fig, use_container_width=True)
                elif graph_type == "Box Plot":
                    fig = px.box(df_release_year, y=parameter, title=f'Distributia {parameter} (Box Plot)')
                    st.plotly_chart(fig, use_container_width=True)

def page_norm_and_stand():
    if 'df' not in st.session_state:
        st.error("""Mergi in meniul "Preluare date din Oracle" si incarca datele!""")
    else:
        df = st.session_state.df.copy(deep=True)
        df['genres'] = df['genres'].apply(lambda x: ', '.join(map(str, x)))
        df['actors'] = df['actors'].apply(lambda x: ', '.join(map(str, x)))

        labelEncoder = LabelEncoder()
        df['title'] = labelEncoder.fit_transform(df['title'].astype(str))
        df['genres'] = labelEncoder.fit_transform(df['genres'].astype(str))
        df['description'] = labelEncoder.fit_transform(df['description'].astype(str))
        df['director'] = labelEncoder.fit_transform(df['director'].astype(str))
        df['actors'] = labelEncoder.fit_transform(df['actors'].astype(str))

        st.subheader("Histograma pentru revenue_millions inainte de standarziare si normalizare")
        df.fillna(df.mean(), inplace=True)
        fig = px.histogram(df, x="revenue_millions", nbins=20)
        st.plotly_chart(fig, use_container_width=True)

        
        df.quantile([.01,.1, .25, .5,.75,.9, .99], axis = 0)
        q_low = df["revenue_millions"].quantile(0.01)
        q_hi  = df["revenue_millions"].quantile(0.99)
        df_filtered = df[(df["revenue_millions"] < q_hi) & (df["revenue_millions"] > q_low)]
        st.subheader("Histograma dupa ce valorile anormale au fost eliminate utilizand quartilele")
        fig = px.histogram(df_filtered, x="revenue_millions", nbins=30)
        st.plotly_chart(fig, use_container_width=True)

        cols=df_filtered.columns.tolist()

        #Scalarea cu StandardScaler
        scaler = StandardScaler()
        scaler.fit(df_filtered)
        df_s = scaler.transform(df_filtered)
        
        df_s = pd.DataFrame(data=df_s, columns=cols)
        st.subheader("Histograma dupa scalare cu StandardScaler")

        fig = px.histogram(df_s, x="revenue_millions", nbins=30)
        st.plotly_chart(fig, use_container_width=True)


        #Scalarea cu MinMax
        scaler = MinMaxScaler()
        scaler.fit(df_filtered)
        df_n = scaler.transform(df_filtered)
        
        df_n = pd.DataFrame(data=df_n, columns=cols)
        st.subheader("Histograma dupa scalare cu MinMax")
        
        fig = px.histogram(df_n, x="revenue_millions", nbins=30)
        st.plotly_chart(fig, use_container_width=True)


        #Normalizarea valorilor
        scaler = Normalizer()
        scaler.fit(df_filtered)
        df_n = scaler.transform(df_filtered)
        
        df_n = pd.DataFrame(data=df_n, columns=cols)
        
        st.subheader("Histograma dupa normalizare cu Normalizer")
        fig = px.histogram(df_n, x="revenue_millions", nbins=30)
        st.plotly_chart(fig, use_container_width=True)

def page_ml():
    st.markdown("""
            ### Importul datelor
            - Datele au fost descarcate de la adresa mentionata in pagina "Introducere" sub format CSV. 
            - Randurile au urmatoarea strucutra: \n
            ```csv
            Rank,Title,Genre,Description,Director,Actors,Year,Runtime (Minutes),Rating,Votes,Revenue (Millions),Metascor
            1,Guardians of the Galaxy,"Action,Adventure,Sci-Fi",A group of intergalactic criminals are forced to work together to stop a fanatical warrior from taking control of the universe.,James Gunn,"Chris Pratt, Vin Diesel, Bradley Cooper, Zoe Saldana",2014,121,8.1,757074,333.13,76
            2,Prometheus,"Adventure,Mystery,Sci-Fi","Following clues to the origin of mankind, a team finds a structure on a distant moon, but they soon realize they are not alone.",Ridley Scott,"Noomi Rapace, Logan Marshall-Green, Michael Fassbender, Charlize Theron",2012,124,7,485820,126.46,65
            3,Split,"Horror,Thriller",Three girls are kidnapped by a man with a diagnosed 23 distinct personalities. They must try to escape before the apparent emergence of a frightful new 24th.,M. Night Shyamalan,"James McAvoy, Anya Taylor-Joy, Haley Lu Richardson, Jessica Sula",2016,117,7.3,157606,138.12,62
            4,Sing,"Animation,Comedy,Family","In a city of humanoid animals, a hustling theater impresario's attempt to save his theater with a singing competition becomes grander than he anticipates even as its finalists' find that their lives will never be the same.",Christophe Lourdelet,"Matthew McConaughey,Reese Witherspoon, Seth MacFarlane, Scarlett Johansson",2016,108,7.2,60545,270.32,59
            5,Suicide Squad,"Action,Adventure,Fantasy",A secret government agency recruits some of the most dangerous incarcerated super-villains to form a defensive task force. Their first mission: save the world from the apocalypse.,David Ayer,"Will Smith, Jared Leto, Margot Robbie, Viola Davis",2016,123,6.2,393727,325.02,40
            ```    
            - Fata de proiectul de la AVMD (sectiunile anterioare), citirea datelor se va realiza direct din fisierul CSV fata de importul si citirea dintr-o baza de date Oracle.
            - Din dropdown se poate alege algoritmul pentru a fi aplicat pe setul de date. Dupa alegere, pasii vor fi executati automat.
            - Ne propunem sa clasificam filmele in filme cu venit mare si filme cu venit mic.
        """)
    
    
    st.markdown("""
        #### Pasi
    """)

    # 1
    st.write("1. Incarcare CSV intr-o variabila dataframe")
    df = pd.read_csv("./movies.csv")
    st.dataframe(df.head())

    #2
    st.write("2. Vizualizare date si analiza descriptiva")
    st.write(df.describe())
    st.write("Valori lipsa")
    st.write(df.isna().sum())

    #3
    st.write("3. Curatarea datelor")
    st.write("3.1 Completare valori lipsa pentru coloanele numerice cu media si pentru coloanele non-numerice cu valoarea cea mai frecventa")
    
    numeric_columns = ['Year', 'Runtime (Minutes)', 'Rating', 'Votes', 'Revenue (Millions)', 'Metascore']
    for col in numeric_columns:
        df[col].fillna(df[col].mean(), inplace=True)

    categorical_columns = ['Genre', 'Director', 'Actors']
    for col in categorical_columns:
        df[col].fillna(df[col].mode()[0], inplace=True)

    st.write("Valori lipsa dupa procesare")
    st.write(df.isna().sum())

    st.write("3.2 Aplicare LabelEncoder pentru coloanele categoriale relevante (Genre, Director, Actors)")
    for col in categorical_columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])

    st.dataframe(df)

    # 4
    st.write("4. Creare coloana binara de clasificare: filme cu venit mare si filme cu venit mic in functie de mediana")
    median_revenue = df['Revenue (Millions)'].median()
    df['High_Revenue'] = (df['Revenue (Millions)'] > median_revenue).astype(int)

    #5
    st.write("5. Selectare features si target (High_Revenue)")
    X = df.drop(columns=['High_Revenue', 'Title', 'Description'])
    y = df['High_Revenue']

    st.dataframe(X)
    st.dataframe(y)

    #6
    st.write("6. Standardizare folosind StandardScaler")
    scaler = StandardScaler()
    X[numeric_columns[:-1]] = scaler.fit_transform(X[numeric_columns[:-1]])  # Exclude 'Revenue (Millions)' from scaling
    X.drop(columns=["Revenue (Millions)"], inplace=True)

    #7
    st.write("7. Dupa standardizare si eliminare 'Revenue (Milions)'")
    st.dataframe(X)

    #8.
    st.write("8. Creare set antrenare (80%) si set testare (20%)")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    #8.
    alghoritms = ["Regresie logistica", "Arbori de decizie", "Random Forest Classifier", "XGB"]

    for alg in alghoritms:
        if alg == "Regresie logistica":
            model = LogisticRegression(max_iter=100)
        elif alg == "Arbori de decizie":
            model = DecisionTreeClassifier()
        elif alg == "Random Forest Classifier":
            model = RandomForestClassifier(random_state=10)
        else:
            model = XGBClassifier(n_jobs=8, random_state = 10)
        
        model.fit(X_train, y_train)

        st.subheader(f"Model {alg}")

        #9.
        st.write(f"Evaluare model {alg}")
        y_pred = model.predict(X_test)

        st.write("Accuracy:", accuracy_score(y_test, y_pred))
        
        st.subheader(f"Matricea de confuzie {alg}")
        conf_matrix = confusion_matrix(y_test, y_pred)
        st.write(conf_matrix)

        st.subheader(f"Raportul de clasificare {alg}")
        st.write(classification_report(y_test, y_pred))

        st.subheader(f"Curba ROC si AUC {alg}")

        from sklearn.metrics import roc_auc_score,roc_curve

        ns_probs = [0 for _ in range(len(y_test))]
        # probabiltatile modelului
        model_probs = model.predict_proba(X_test)
        # pstram doar probabilitatile pentru valorile pozitive
        model_probs = model_probs[:, 1]
        # calcul scor auc
        ns_auc = roc_auc_score(y_test, ns_probs)
        lr_auc = roc_auc_score(y_test, model_probs)
        st.write('No Skill: ROC AUC=%.3f' % (ns_auc))
        st.write('ROC AUC=%.3f' % (lr_auc))
        #  roc curves
        ns_fpr, ns_tpr, _ = roc_curve(y_test, ns_probs)
        model_fpr, model_tpr, _ = roc_curve(y_test, model_probs)
        # plot the roc curve for the model
        fig, ax = plt.subplots(figsize=(8, 6))

        ax.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')
        ax.plot(model_fpr, model_tpr, marker='.', label='Clasifier')
        # axis labels
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.legend()

        st.pyplot(fig)
    
def main():

    nav = st.sidebar.radio("Pagini", ["Aplicare algoritmi Machine Learning (ML)", "Introducere (AVMD)" , "Importul datelor (AVMD)", "Preluare date din Oracle & preprocesare (AVMD)", "Standardizare si normalizare (AVMD)", "Analiza descriptiva (AVMD)", "Analiza vizuala (grafice) (AVMD)"])

    if nav != "Aplicare algoritmi Machine Learning (ML)" or nav is None:
        st.title("Proiect AVMD Tarba Sabin")
    else:
        st.title("Proiect ML Tarba Sabin")

    if nav == "Introducere (AVMD)":
        page_intro()
    elif nav == "Importul datelor (AVMD)":
        page_data_import()
    elif nav == "Preluare date din Oracle & preprocesare (AVMD)":
        page_get_data_from_db()
    elif nav == "Standardizare si normalizare (AVMD)":
        page_norm_and_stand()
    elif nav == "Analiza descriptiva (AVMD)":
        page_descriptive_analysis()
    elif nav == "Analiza vizuala (grafice) (AVMD)":
        page_visual_analysis()
    elif nav == "Aplicare algoritmi Machine Learning (ML)":
        page_ml()

if __name__ == "__main__":
    main()