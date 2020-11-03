#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 29 19:28:24 2020

@author: swift
"""
import numpy as np
import pickle
import pandas as pd

from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, ColorBar, NumeralTickFormatter, Title
from bokeh.models.callbacks import CustomJS
from bokeh.models.tools import HoverTool
from bokeh.io import  output_notebook
from bokeh.layouts import row, widgetbox,  column
from bokeh.models.widgets import  Select
from bokeh.palettes import Spectral11, Spectral3
from bokeh.transform import linear_cmap , dodge
from bokeh.io import  show
from bokeh.models.widgets import AutocompleteInput,  Button, Paragraph
from bokeh.application import Application
from bokeh.application.handlers import FunctionHandler

from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import ComplementNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression



    

def javaScript():
     return """
            
            var data = source.data;
            var selectV = cb_obj.value;
            var index =  colDS.data['DE'].indexOf(selectV);
            var f = colDS.data['index'][index];

            if (cb_obj.title === 'X Achse'){
                data['x']  = allsources.data[f];
                xaxis.axis_label=colDS.data['Axis'][index];
            } else if(cb_obj.title === 'Y Achse'){
                data['y']  = allsources.data[f];
                yaxis.axis_label=colDS.data['Axis'][index];
                            
            } else {
                var min_max_data = min_max.data;
                mapper['transform'].low=min_max_data[f][0];
                mapper['transform'].high=min_max_data[f][1];
                data['z'] = allsources.data[f];
                hover.tooltips[2] = [selectV, '@z']
                color_bar.color_mapper = mapper['transform'];
                subtitle.text = 'mit Einfärbung ' + colDS.data['Axis'][index];
                if(min_max_data[f][1] === 0){
                        color_bar.visible = false;
                        subtitle.text = colDS.data['Axis'][index] + ' ist bei allen Filmen 0';
                } else {
                        color_bar.visible = true;
                }
                    console.log(color_bar);
            }
            title.text = 'Gegenüberstellung ' +  xaxis.axis_label + ' vs ' + yaxis.axis_label 
            source.change.emit();
            
        """
    
    
def loadGraphic(df,integerColumns ):

    output_notebook()

    colValues = pd.read_json (r"""JSONData/Bezeichnungen.json""")
    defaultZDimension = 'WinOscar'
    defaultXAxis = 'Runtime'
    defaultYAxis = 'TomatoRating'
    selectAxisOptions = [ colValues.loc[x]['DE'] for x in colValues.index.tolist() if 'Win' not in x and 'Nomination' not in x ]
    colorOptions = list(colValues['DE'])
    
    #DataSources erstellen
    colValuesDS = ColumnDataSource(colValues)
    min_max_DF = ColumnDataSource(df[integerColumns].describe().loc[['min','max']])
    source = ColumnDataSource(dict(x=df[defaultXAxis], y=df['imdbRating'], z=df[defaultZDimension],Title=df['Title'], Year=df['Year'] ))
    allsource = ColumnDataSource(df[integerColumns])
    mapper = linear_cmap(field_name='z', palette=Spectral11 ,low=min(df[defaultZDimension]) ,high=max(df[defaultZDimension]))
    
    #Select Elemente erstellen
    selectX = Select(title = 'X Achse',height = 50,  options = selectAxisOptions , value=colValues.loc[defaultXAxis]['DE'])
    selectY = Select(title = 'Y Achse',height = 50,  options = selectAxisOptions, value=colValues.loc[defaultYAxis]['DE'])
    selectC = Select(title = 'Einfärbung',height = 50,  options = colorOptions, value=colValues.loc[defaultZDimension]['DE'])
    
    # Formatter
    formatter = NumeralTickFormatter(format='0.0a')
   
    
    #Erstellung des Plot
    p = figure()
    p.circle(x='x', y='y',source=source,size=10, line_color=mapper,color=mapper)
    p.xaxis.axis_label = colValues.loc[defaultXAxis]['Axis']
    p.yaxis.axis_label = colValues.loc[defaultYAxis]['Axis']
    p.xaxis.formatter =formatter
    p.yaxis.formatter =formatter
    
        
    #Hover erstellen 
    hover = HoverTool()
    hover.tooltips=[('Name', '@Title'),('Jahr', '@Year'),(colValues.loc[defaultZDimension]['DE'], '@z')]
    p.add_tools(hover)
    
    
    #Title erstellen 
    subtitle = Title(text='mit Einfärbung ' + hover.tooltips[2][0], text_font_style="italic")
    title = Title(text='Gegenüberstellung ' +  colValues.loc[defaultXAxis]['Axis'] + ' vs ' + colValues.loc[defaultYAxis]['Axis'] , text_font_size="10pt")
    p.add_layout(subtitle, 'above')
    p.add_layout(title, 'above')
    
    #Color Bar erstellen 
    color_bar = ColorBar(color_mapper=mapper['transform'], width=8,  location=(0,0), label_standoff=12, formatter = formatter )
    p.add_layout(color_bar, 'right')
    
    
        # Callback Funktion
    callback = CustomJS(args=dict(hover = hover, allsources = allsource,title = title,
                                  source=source,color_bar=color_bar,min_max = min_max_DF,
                                  yaxis=p.yaxis[0], xaxis=p.xaxis[0], mapper = mapper, 
                                  colDS = colValuesDS, subtitle = subtitle ), 
                                  code=javaScript())
    
    # Callback Funktion übergeben 
    selectX.callback = callback
    selectY.callback = callback
    selectC.callback = callback
    
    return row(p,row(widgetbox(selectX,selectY, selectC)))


def js():  
    return """
        var data = source.data;
        var predata = presource.data;
        var from = parseInt(fromYear.value)
        var to = parseInt(toYear.value)
        
        if(from < to){
            dimension = sort.value + '_' + fromYear.value + '_' + toYear.value  + '_' + value.value;

            console.log(dimension);
            source.data['akteur'] = predata[dimension][2];
            source.data['Metascore'] = predata[dimension][0];
            source.data['imdbRating'] = predata[dimension][3];
            source.data['TomatoRating'] = predata[dimension][1];
            xr.factors = predata[dimension][2];
            console.log(source);
            source.change.emit();
            reset.data['fromYear'] = fromYear.value;
            reset.data['toYear'] = toYear.value;
        } else {
            fromYear.value = reset.data['fromYear']
            toYear.value  = reset.data['toYear']
        }
        yaxis.axis_label = "Top 10 "+value.value+" sortiert nach "+sort.value+" von oben nach unten"
        title.text = "Bewertung der Top 10 " +  value.value+ " aus den Jahren " +fromYear.value+ " bis "+toYear.value 
        """

def loadGraphic2():
    
    data = pd.read_json (r"""JSONData/barPlotData.json""")
    types =  ['Actors', 'Genre', 'Director', 'Title', 'Production']
    years1 =  [str(i) for i in range(1980,2020,1)]
    sort =  ['Gesamt Durchschnitt', 'Metascore','imdbRating','TomatoRating']
    
    output_notebook()
    
    typeDict ={              'akteur' : data['Gesamt Durchschnitt_1980_2019_Actors']['akteur'],
                            'Metascore'   : data['Gesamt Durchschnitt_1980_2019_Actors']['Metascore'],
                            'imdbRating'   : data['Gesamt Durchschnitt_1980_2019_Actors']['imdbRating'],
                            'TomatoRating'   : data['Gesamt Durchschnitt_1980_2019_Actors']['TomatoRating']}
    resetSource = pd.DataFrame()
    resetSource['fromYear'] = '1980'
    resetSource['toYear'] = '2019'
    mainSource = ColumnDataSource(data=data)
    resetSource = ColumnDataSource(data=resetSource)
    source = ColumnDataSource(data=typeDict)
    
    selectW = Select(title = 'Sortierung nach:',height = 50,  options = sort, value = "Gesamt Durchschnitt" )
    selectX = Select(title = 'Dimension:',height = 50,  options = types, value = "Actors" )
    selectY = Select(title = 'Jahr von:',height = 50,  options = years1, value = "1980" )
    selectZ = Select(title = 'Jahr bis:',height = 50,  options = years1, value =  "2019")
    
    p = figure(y_range=typeDict['akteur'], x_range=(0, 1.25), plot_height=450, plot_width=850,
           toolbar_location=None, tools="")

    p.hbar(y=dodge('akteur', -0.25, range=p.y_range), right='Metascore', height=0.2, source=source,
           color=Spectral3[0], legend ="Meta")

    p.hbar(y=dodge('akteur',  0.0,  range=p.y_range), right='imdbRating', height=0.2, source=source,
           color=Spectral3[1], legend ="IMDB")

    p.hbar(y=dodge('akteur',  0.25, range=p.y_range), right='TomatoRating', height=0.2, source=source,
           color=Spectral3[2], legend ="Tomato" )

    title = Title(text="Durchschnitt Bewertung  der Top 10 Schauspieler aus den Jahren 1980 bis 2019", text_font_size="10pt")
    p.add_layout(title, 'above')
    p.xaxis.axis_label = "Bewertungen in %"
    p.yaxis.axis_label = "Top 10 Schauspieler sortiert von oben nach unten"
    p.xgrid.grid_line_color = None

    p.legend.location = "top_right"

   


    callback = CustomJS(args=dict(source=source, 
                                  presource = mainSource,
                                  xr=p.y_range,
                                  reset = resetSource,
                                  fromYear =selectY , 
                                  toYear = selectZ, 
                                  sort = selectW , 
                                  value = selectX,
                                  title = title, 
                                  yaxis=p.yaxis[0], xaxis=p.xaxis[0]
                                  ),  code=js())



    selectX.callback = callback
    selectY.callback = callback
    selectZ.callback = callback
    selectW.callback = callback


    show(column(column(column(row(selectX,selectY),row(selectW,selectZ ) ), p)))


def addSubPlot(scoreType, axis, df):
    binsDimensions =  np.arange(0, 1.05, 0.05).tolist()
    
    # Generate normend Histogramm
    weights = np.ones_like(df[scoreType])/float(len(df[scoreType]))
    n, bins, patches = axis.hist(x=df[scoreType], weights=weights, density=False, bins= binsDimensions , color='#0504aa',edgecolor='black')
    
    # get 1 and 3 Quantil                             
    q1, q3 =  np.percentile(df[scoreType], 25), np.percentile(df[scoreType], 75)
    axis.axvline(x=q1, color='Orange', linestyle='--', label = '25% Quantil: ' + str(q1))
    axis.axvline(x=q3, color='Red', linestyle='--', label = '75%. Quantil: ' + str(q3))
    
    
    axis.set(xlabel='Bewertung', xticks = binsDimensions)
    axis.set_title(scoreType)
    axis.legend()
    
    for tick in axis.get_xticklabels():
        tick.set_rotation(90)
        
    return q1, q3



def checkClassifikationMethods(df, testSystem = 'BernoulliNB' ):

        
    directorsDF_ohe, directorList = get_one_hot_ecnodet_ActorsNew(df, 'Director')
    writersDF_ohe, writersList = get_one_hot_ecnodet_ActorsNew(df, 'Writer')
    productionDF_ohe, productionList = get_one_hot_encodet_Production(df)
    genreDF_ohe, genreList = get_one_hot_ecnodet_ActorsNew(df, 'Genre',20)
    actorsDF_ohe, actorList = get_one_hot_ecnodet_ActorsNew(df, 'Actors',4)
    
    data_frames = [genreDF_ohe, directorsDF_ohe,writersDF_ohe, actorsDF_ohe, productionDF_ohe  ]
    df_merged = mergeDataFrames(data_frames)
        
    df_merged = addCategorys(df_merged , df , 'Metascore', getRatingQuantile(df))
    
    X = df_merged.loc[:, df_merged.columns != 'Category']
    y = df_merged['Category']
        
    result_df = pd.DataFrame(0.0, index=  np.arange(1, 11, 1),columns=['TrainScore',
                                'TestScore',
                                'BAD_Precision', 'BAD_Recall', 'BAD_f1score' ,
                                'AVG_Precision', 'AVG_Recall', 'AVG_f1score' ,
                                'TOP_Precision', 'TOP_Recall', 'TOP_f1score' 
                                 ] )
        
    for param in np.arange(1, 11, 1):
        print(param)
        X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle = True, test_size = 0.3)
        
        if testSystem == 'BernoulliNB' :
            tree =BernoulliNB(alpha= param/10)
        elif testSystem == 'GaussianNB' :
            tree = GaussianNB(var_smoothing = param/10)
        elif testSystem == 'MultinomialNB' :
            tree = MultinomialNB(alpha = param/10)
        elif testSystem == 'ComplementNB' :
            tree = ComplementNB(alpha = param/10)
        elif testSystem == 'DecisionTreeClassifier' :
            tree = DecisionTreeClassifier(criterion= 'entropy', max_depth = param)
            #tree = DecisionTreeClassifier(criterion= 'gini', max_depth = param)
        else : 
            tree = KNeighborsClassifier(neighbors =param)
        
        tree.fit(X_train, y_train)
        y_pred = tree.predict(X_test)   
            
        classiReport = classification_report(y_test, y_pred, output_dict = True)

        result_df.at[param, 'TrainScore'] =   round(tree.score(X_train,y_train),3)
        result_df.at[param, 'TestScore']  =  round(tree.score(X_test,y_test),3)
        result_df.at[param, 'AVG_Precision'] = round(classiReport['Durchschnittlicher Film']['precision'],3)
        result_df.at[param, 'AVG_Recall'] = round(classiReport['Durchschnittlicher Film']['recall'],3)
        result_df.at[param, 'AVG_f1score'] = round(classiReport['Durchschnittlicher Film']['f1-score'],3)
        result_df.at[param, 'TOP_Precision'] = round(classiReport['Guter Film']['precision'],3)
        result_df.at[param, 'TOP_Recall'] = round(classiReport['Guter Film']['recall'],3)
        result_df.at[param, 'TOP_f1score'] = round(classiReport['Guter Film']['f1-score'],3)
        result_df.at[param, 'BAD_Precision'] = round(classiReport['Schlechter Film']['precision'],3)
        result_df.at[param, 'BAD_Recall'] = round(classiReport['Schlechter Film']['recall'],3)
        result_df.at[param, 'BAD_f1score'] = round(classiReport['Schlechter Film']['f1-score'],3)

    return result_df





def getRatingQuantile(df):
    ratingQuantile = {}
    ratingQuantile['Metascore'] = calculateQuantile('Metascore', df)
    ratingQuantile['imdbRating'] = calculateQuantile('imdbRating', df)
    ratingQuantile['TomatoRating'] = calculateQuantile('TomatoRating', df)
    return ratingQuantile


def calculateQuantile(scoreType,  df):
    import numpy as np                              
    return  np.percentile(df[scoreType], 25), np.percentile(df[scoreType], 75)
    

def addCategorys(df_input,df, metric,quantileList):
    import numpy as np 
    conditions = [
    (df[metric] >= quantileList[metric][1]),
    (df[metric] <= quantileList[metric][0])]
    
    choices = ['Guter Film', 'Schlechter Film']
    
    df_input['Category'] = np.select(conditions, choices, default='Durchschnittlicher Film')
    return df_input


def get_one_hot_ecnodet_ActorsNew(df, actors_or_directors, n = 5):
    import pandas as pd  
    prefix = actors_or_directors + '_'
    
    all_actors_or_directors_liste = list(map(lambda x: list(set(x))[:n] , df[actors_or_directors]))
    flat_list = list(set([item for sublist in all_actors_or_directors_liste for item in sublist]))
    
    actors_or_directors_df = pd.DataFrame()
    seriesListe = list(map(lambda x: pd.Series(1, index = list(set(x))[:n] ) , df[actors_or_directors]))
    actors_or_directors_df  = actors_or_directors_df .append(seriesListe) 
    actors_or_directors_df = actors_or_directors_df.add_prefix(prefix)
    actors_or_directors_df = actors_or_directors_df.set_index([pd.Index( df.index.tolist())])
    return actors_or_directors_df, flat_list



def get_one_hot_encodet_Production(df):
    import pandas as pd  
    productionDF_ohe = pd.get_dummies(df['Production'],prefix='Production_')
    productionlist = list(set(df['Production'].tolist()))
    return productionDF_ohe, productionlist

def mergeDataFrames(data_frames):
    import pandas as pd  
    from functools import reduce
    mergedDataFrame=  reduce(lambda  left,right: pd.merge(left,right,left_index=True, right_index=True), data_frames)
    return   mergedDataFrame.fillna(0)





def printf1ScoreByIncreasingDimensions():
    import pandas as pd  
    import matplotlib.pyplot as plt
    
    result = pd.read_json (r"""JSONData/DimensionReductionTest.json""")
    
    
    
    f1Scores = ['Accuracy', 'Durchschnitt F1-Score', 'Schlecht F1-Score', 'Gut F1-Score']
    recallScores = ['Accuracy', 'Durchschnitt Recall', 'Schlecht Recall', 'Gut Recall']
    precisionScores = ['Accuracy', 'Durchschnitt Precision', 'Schlecht Precision', 'Gut Precision']
    result.index = result['Dimensionen']
    result = result.sort_index()
    
    fig, axs = plt.subplots(2, 2, figsize=(15,15))
    
    axs[0, 0].plot(result.index , result[f1Scores])
    axs[0, 0].set_title('F1 Score Entwicklung')
    axs[0, 0].legend(f1Scores)
    axs[0, 0].set_xlabel('Anzahl der Dimensionen')
    axs[0, 0].set_ylabel('Metric in %')
    
    
    axs[0, 1].plot(result.index , result[recallScores])
    axs[0, 1].set_title('Recall Entwicklung')
    axs[0, 1].legend(recallScores)
    axs[0, 1].set_xlabel('Anzahl der Dimensionen')
    axs[0, 1].set_ylabel('Metric in %')
    
    axs[1, 0].plot(result.index , result[precisionScores])
    axs[1, 0].set_title('Precision Entwicklung')
    axs[1, 0].legend(precisionScores)
    axs[1, 0].set_xlabel('Anzahl der Dimensionen')
    axs[1, 0].set_ylabel('Metric in %')
    
    axs[1, 1].plot(result.index, result['Dimensionen/DatasetSize'], 'tab:red')
    axs[1, 1].set_title('Ratio Anzahl Dimensionen / Datensatzgröße')
    axs[1, 1].set_xlabel('Anzahl der Dimensionen')
    axs[1, 1].set_ylabel('Verhältnis Anzahl Dimensionen gegenüber Datensatzgröße')
    
    plt.show()


def multivarianter_Regressions_Test(df,genreDF_ohe, directorsDF_ohe, writersDF_ohe,actorsDF_ohe , productionDF_ohe, ratingQuantile):

    score= 'Metascore'
        
    data_frames = [ genreDF_ohe, directorsDF_ohe,writersDF_ohe, actorsDF_ohe, productionDF_ohe  ]
    df_merged = mergeDataFrames(data_frames)      
    df_merged = addCategorys(df_merged , df , score,ratingQuantile)
        
    X_ = df_merged.loc[:, df_merged.columns != 'Category']
    y_ = df[score]
    
    #Diesen Code für LogisticRegression aktivieren
    #y_ = df_merged['Category']
    
    
    result_df = pd.DataFrame(0.0, index=  np.arange(0,1, 1),columns=['R^2TrainScore',
                                'R^2TestScore','Accuracy',
                                'BAD_Precision', 'BAD_Recall', 'BAD_f1score' ,
                                'AVG_Precision', 'AVG_Recall', 'AVG_f1score' ,
                                'TOP_Precision', 'TOP_Recall', 'TOP_f1score' 
                                 ] )
    
    
    
    for testrun in range(1,10) :
        # Erstellung der Test und Trainingsdaten
        X_train, X_test, y_train, y_test = train_test_split(X_, 
                                                        y_,  
                                                        test_size = 0.3, 
                                                        shuffle = True)
    
        for param in np.arange(0, 1, 1):
            
            
            #Auswahl des Verfahrens 
            #mir = DecisionTreeRegressor(max_depth = param)
            mir = LinearRegression()
            #mir = linear_model.BayesianRidge()
            #mir = LogisticRegression(random_state=param, solver = 'saga', multi_class= 'ovr')
            mir.fit(X_train,y_train )

            
            result_df.at[param, 'R^2TrainScore'] = result_df.loc[param]['R^2TrainScore'] + mir.score(X_train, y_train)
            result_df.at[param, 'R^2TestScore']  = result_df.loc[param]['R^2TestScore'] + mir.score(X_test,y_test)
            
            y_test =  y_test.rename('Expected_Values')
            resultDF = pd.DataFrame(y_test);
            resultDF['Predicted_Values'] = mir.predict(X_test)
            classiReport = checkResults(resultDF,ratingQuantile, score )

            
            result_df.at[param, 'Accuracy'] = result_df.loc[param]['Accuracy'] + classiReport['accuracy']
            result_df.at[param, 'AVG_Precision'] = result_df.loc[param]['AVG_Precision'] + classiReport['Durchschnittlicher Film']['precision']
            result_df.at[param, 'AVG_Recall'] = result_df.loc[param]['AVG_Recall'] + classiReport['Durchschnittlicher Film']['recall']
            result_df.at[param, 'AVG_f1score'] = result_df.loc[param]['AVG_f1score'] + classiReport['Durchschnittlicher Film']['f1-score']
            result_df.at[param, 'TOP_Precision'] = result_df.loc[param]['TOP_Precision'] + classiReport['Guter Film']['precision']
            result_df.at[param, 'TOP_Recall'] =  result_df.loc[param]['TOP_Recall'] +classiReport['Guter Film']['recall']
            result_df.at[param, 'TOP_f1score'] =  result_df.loc[param]['TOP_f1score'] +classiReport['Guter Film']['f1-score']
            result_df.at[param, 'BAD_Precision'] =  result_df.loc[param]['BAD_Precision'] +classiReport['Schlechter Film']['precision']
            result_df.at[param, 'BAD_Recall'] =  result_df.loc[param]['BAD_Recall'] +classiReport['Schlechter Film']['recall']
            result_df.at[param, 'BAD_f1score'] =  result_df.loc[param]['BAD_f1score'] +classiReport['Schlechter Film']['f1-score']
        
        
    # Berechnung des arithm. Mittel der Train- und Test-Scores
    result_df = result_df.div(testrun)   
    result_df= result_df.round(decimals=3)
        
    return result_df





def checkResults(dfUebergabe,quantileList, metric ):

    
    #Diesen Code für LogisticRegression aktivieren
    #classiReport = classification_report(dfUebergabe['Predicted_Values'], dfUebergabe['Expected_Values'], output_dict = True)
    #return classiReport

    conditions_Predicted = [
    (dfUebergabe['Predicted_Values'] >= quantileList[metric][1]),
    (dfUebergabe['Predicted_Values'] <= quantileList[metric][0])]
    conditions_Expected = [
    (dfUebergabe['Expected_Values'] >= quantileList[metric][1]),
    (dfUebergabe['Expected_Values'] <= quantileList[metric][0])]
    
    choices = ['Guter Film', 'Schlechter Film']
    
    dfUebergabe['Predicted_Category'] = np.select(conditions_Predicted, choices, default='Durchschnittlicher Film')
    dfUebergabe['Expected_Category'] = np.select(conditions_Expected, choices, default='Durchschnittlicher Film')
    classiReport = classification_report(dfUebergabe['Predicted_Category'], dfUebergabe['Expected_Category'], output_dict = True)
    return classiReport

def beliebtheitsermittler(df,actorList, productionList , directorList, writersList, genreList,df_merged, ratingQuantile ):

    text =  "Füllen Sie die Felder aus und drücken auf den Start-Button um den Film zu bewerten."
    actorList = list(set(actorList))
    productionList = list(set(productionList))
    directorList = list(set(directorList))
    writersList = list(set(writersList))
    genreList = list(set(genreList))
    
    
    output_notebook()
    
    # Set up widgets
    actor1 = AutocompleteInput(completions=actorList, placeholder = 'Required',title='Schauspieler 1')
    actor2 = AutocompleteInput(completions=actorList, placeholder = 'Required',  title='Schauspieler 2')
    actor3 = AutocompleteInput(completions=actorList, placeholder = 'Required',  title='Schauspieler 3')
    actor4 = AutocompleteInput(completions=actorList, placeholder = 'Required',  title='Schauspieler 4')
    production = AutocompleteInput(completions=productionList, placeholder = 'Required', title='Production')
    director = AutocompleteInput(completions= directorList, placeholder = 'Required', title='Regie')
    autor = AutocompleteInput(completions= writersList, placeholder = 'Required', title='Autor')
    genre = Select(title = 'Auswahl des Hauptgenre',height = 50,value = 'Comedy',  options = genreList)
    
    select = Select(title = 'Auswahl der Metric',height = 50,value = 'Metascore',   options = ['Metascore', 'imdbRating', 'TomatoRating'])
    answer = Button(height = 100, width = 600,disabled = True,margin = [10,10,10,10],background = 'white', label = text)
    button = Button(margin = [23,0,0,200], width = 100, button_type = 'primary', label = 'Start')
    paragraph2 = Paragraph(margin = [40,0,0,10])
    paragraph2.text = "Ergebnis:"
    
    
    def doOnClick():
        metric = select.value
        columns = df_merged.columns.tolist()
        columns.remove('Category')
        
        if  (actor1.value in actorList) and (actor2.value in actorList) and (actor3.value in actorList) and (actor4.value in actorList) and (production.value in productionList) and (director.value in directorList) and (autor.value in writersList):
            filename = 'models/'+ metric + '_model.sav'
            
            model = pickle.load(open(filename, 'rb'))
            d = pd.DataFrame(0,index=np.arange(1), columns=columns)
            
            d['Actors_' + actor1.value] = 1
            d['Actors_' + actor2.value] = 1
            d['Actors_' + actor3.value] = 1
            d['Actors_' + actor4.value] = 1
            d['Genre_' + genre.value] = 1
            d['Director_' + director.value] = 1
            d['Writer_' + autor.value] = 1
            d['Production__'+ production.value]  = 1  
            
            result = model.predict(d)[0]

            if ('Gut' in result):
                answer.background = 'green'

            elif('Schlecht' in result):
                answer.background = 'red'
    
            else:
                answer.background = 'yellow'

             
            answer.label = result
        else:
            answer.label = text
            answer.background = 'white'
        
          
    button.on_click(doOnClick)
    
    
        
    layout = [row(widgetbox(actor1 , actor2, actor3, actor4, genre),
                  widgetbox( production, director,autor, select, button ))]
    
    def modify_doc(doc):

        doc.add_root(row(layout))
        doc.add_root(column(widgetbox(paragraph2,answer)))
    
    handler = FunctionHandler(modify_doc)
    app = Application(handler)
    show(app)

def createModels(df,df_merged ,ratingQuantile):
    import pickle
    from sklearn.naive_bayes import BernoulliNB
    scores = ['TomatoRating', 'Metascore','imdbRating']
    for score in scores:
        df_merged = df_merged
        df_merged = addCategorys(df_merged , df , score,ratingQuantile)
        model = BernoulliNB(alpha = 0.6)
        X_ = df_merged.loc[:, df_merged.columns != 'Category']
        y_ = df_merged['Category']
        model.fit(X_, y_)
        filename ='models/' +  score+ '_model.sav'
        pickle.dump(model, open(filename, 'wb'))
