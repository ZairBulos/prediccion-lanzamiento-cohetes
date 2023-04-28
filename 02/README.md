# Recopilación y manipulación de datos 📂🗃️

Obtenga información sobre los pasos para importar datos en Python y limpiarlos para usarlos en la creación de modelos de Machine Learning.

### Objetivos de aprendizaje:

* Explorar datos sobre el tiempo en los días en que se lanzaron cohetes tripulados y no tripulados.
* Explorar datos meteorológicos de los días cercanos a los lanzamientos.
* Limpiar los datos como preparación para entrenar el modelo de Machine Learning.

<hr/>

## Introducción

Cada día se realiza el seguimiento del tiempo y se analiza para ayudar a los aviones a tener vuelos seguros. Se deben supervisar muchas condiciones meteorológicas para garantizar que la probabilidad de que se produzca algo negativo en el avión sea lo más baja posible. Con el lanzamiento de cohetes, el riesgo y los resultados de una lectura equivocada o de un fragmento de datos sin seguimiento pueden resultar devastadores.

A mayor altitud, más potencia y la incapacidad de controlar la dirección del cohete, la precisión de las predicciones de clima seguro es una de las partes más complicadas e importantes de la exploración espacial. Las complejidades aumentan incluso más cuando se tiene en cuenta la escala de tiempo. Las fechas de lanzamiento se deciden con años de antelación para poder dar cabida a enormes esfuerzos de preparación y programación.

No tenemos la experiencia en la materia ni acceso a tantos datos meteorológicos como la NASA. Sin embargo, en este módulo se le ofrecerá información sobre cómo usar datos meteorológicos que se pueden obtener fácilmente para simular el enfoque que toman los científicos de la NASA para abordar el mismo problema. En este módulo se le enseña cómo preparar datos meteorológicos básicos para entrenar un modelo de Machine Learning a fin de predecir cuáles son los días adecuados para el lanzamiento de cohetes.

<hr/>

## Determinación de las preguntas que se deben formular sobre el lanzamiento de cohetes

La ciencia de datos es un proceso iterativo entre:

* El conocimiento y la comprensión de lo que es hoy en día.
* Los datos que se han recopilado.
* Las preguntas que se están haciendo.

Las preguntas nuevas dan paso a más información y a la intención de recopilar más datos.

Cuando se planea una nueva misión, los científicos de la NASA tienen que preguntarse: "¿Qué día, dentro de X número de años, tendrá menos probabilidades de requerir una fuerza adicional para el lanzamiento debido a las condiciones meteorológicas?" En los días previos al lanzamiento del cohete, los científicos de la NASA son los más críticos al preguntarse: "¿Causará el tiempo en esta zona en este momento algún problema potencial para el lanzamiento?".

Para responder a estas preguntas, la NASA cuenta con expertos en cohetes, meteorología y aviación que diseñan instrucciones y modelos que les ayudan a tomar una determinación. También tienen datos de sus propios sensores y globos meteorológicos, así como fuentes de confianza como la [National Oceanic and Atmospheric Administration (NOAA, Oficina Nacional de Administración Oceánica y Atmosférica)](https://www.noaa.gov/).

En este módulo, no se dispone de todos los datos o conocimientos que la NASA tiene sobre el día de un lanzamiento, pero sí de datos meteorológicos sencillos disponibles públicamente. En este módulo se examinará lo siguiente:

* Condiciones (nubosidad, nubosidad parcial, despejado, lluvia, truenos, tormentas)
* Temperatura
* Humedad
* Velocidad del viento
* Dirección del viento
* Precipitación
* Visibilidad
* Nivel de mar
* Presión

En esta ruta de aprendizaje, usará la inteligencia artificial y el aprendizaje automático para detectar patrones meteorológicos en días en los que se lanzaron cohetes correctamente. Con esos patrones, predecirá si es probable que se produzca un lanzamiento en función de condiciones meteorológicas específicas.

### Desafío adicional

En este módulo se le guía a través de una forma concreta de resolver este problema. Ahora puede detenerse un momento para realizar predicciones y en pensar en otros datos o en preguntas que podría formular sobre la seguridad de los lanzamientos de cohetes.

Por ejemplo, ¿cree que la temperatura es un indicador más importante de la seguridad de los lanzamientos que las precipitaciones? ¿Puede usar [Azure Cognitive Services](https://azure.microsoft.com/es-es/products/cognitive-services/) para tomar imágenes de satélite en tiempo real y usar la clasificación de imágenes para determinar los tipos de nubes y su relación con la probabilidad de que un lanzamiento sea seguro?

<hr/>

## Exploración de los datos de lanzamiento de cohetes para entenderlos

Los modelos de Machine Learning se entrenan con datos suficientes para evitar errores. Sin datos suficientes, es posible que un modelo de Machine Learning sea demasiado general.

Por ejemplo, si ha entrenado un modelo de Machine Learning con datos de temperatura, es posible que no detecte que la precipitación es más importante y que no siempre se correlaciona con temperaturas bajas en Florida, Estados Unidos. En ese caso, el modelo podría indicar que es seguro lanzar un cohete en un día con una temperatura buena pero demasiadas precipitaciones, lo cual no sería seguro.

### Recopilación de datos

El primer paso en cualquier solución de ciencia de datos o aprendizaje automático consiste en recopilar y comprender los datos. En esta ruta de aprendizaje, se han recopilado datos meteorológicos disponibles de forma pública en [NOAA](https://www.noaa.gov/) y [Weather Underground](https://www.wunderground.com/history) para las fechas de lanzamiento de cohetes de la NASA tomadas de la [lista de misiones de la NASA en Wikipedia](https://wikipedia.org/wiki/List_of_NASA_missions). Después, compilamos estos datos en un archivo de Excel.

El archivo de Excel contiene los datos meteorológicos de los días de lanzamientos individuales tripulados y no tripulados. También se han agregado datos de los dos días que rodean a los lanzamientos para comprobar si hay algún patrón interesante. Esta es una captura de pantalla del archivo de Excel.

![excel](https://learn.microsoft.com/es-es/training/modules/collect-manipulate-data-python-nasa/media/excel.png)

### Datos que faltan

El archivo de Excel tiene abundantes datos sobre cada lanzamiento. Pero a medida que empiece a explorar estos datos, es posible que encuentre un problema importante. Solo una fila representa un lanzamiento de cohete que supuestamente debía haberse realizado pero que se aplazó debido a problemas meteorológicos:

Fila 294: Space X Dragon; 27 de mayo de 2020

Una lista de todos los lanzamientos intentados pero pospuestos debido al tiempo no es tan fácil de detectar como la lista de lanzamientos correctos. Las fechas que se tuvieron en cuenta pero que se cambiaron antes del anuncio de la fecha prevista para el lanzamiento tampoco son fáciles de detectar.

### Expertos en la materia

El [45th Space Wing de las Fuerzas Aéreas de Estados Unidos](https://www.patrick.af.mil/About-Us/Weather/) tiene una misión: "Aprovechar el clima para garantizar un acceso *seguro* al aire y al espacio". En combinación con las mentes de la NASA, la probabilidad de elegir una fecha que se vea afectada por problemas meteorológicos es pequeña. Para garantizar el menor número de cambios en la programación de un lanzamiento, los expertos en meteorología y aviación tienen en cuenta los cambios climáticos, los patrones meteorológicos y los datos existentes.

Puede empezar a explorar este problema por su cuenta si visita la [programación de lanzamientos de la NASA](https://www.nasa.gov/launchschedule/). Incluso sin aprendizaje automático, eche un vistazo a los patrones meteorológicos previstos en Cabo Cañaveral. Trate de identificar por qué se eligieron una fecha y hora determinadas en vez de otras una semana antes o después.

### Búsqueda de datos adicionales

El objetivo de esta ruta de aprendizaje es que emprenda un curioso recorrido por el tiempo y su relación con los lanzamientos de cohetes. Le recomendamos que descubra más datos para mejorar un modelo de Machine Learning propio. Es parte del recorrido por la ciencia de datos.

¿Qué cree que podría usar para detectar lanzamientos que han tenido que retrasarse debido al tiempo? ¿Artículos de noticias? ¿Archivos?

<hr/>

## Ejercicio: Importación de bibliotecas de Python y datos de lanzamiento de cohetes

Ahora tiene un objetivo: *¿Es probable que se produzca un lanzamiento dadas las condiciones meteorológicas específicas?* Tiene un conjunto de datos que contiene datos meteorológicos de:

* Varios lanzamientos correctos
* Un día de lanzamiento aplazado
* Los días anteriores y posteriores a cada lanzamiento.

Ahora puede empezar a programar.

### Aprendizaje automático en el código

Puede usar diversas herramientas y servicios para solucionar problemas de aprendizaje automático. En estas rutas de aprendizaje sobre el espacio se usa Visual Studio Code, Python, scikit-learn y Azure.

Vea este vídeo de Microsoft para obtener información sobre cómo descargar y configurar un entorno similar al que necesitará.

![video](https://learn.microsoft.com/video/media/2ebb2148-b934-49e4-a710-4fb45882fddc/howdoyousetupyourlocalenvironmentfordataexplorati_960.jpg)

Al configurar el entorno de programación local, se recomienda crear un entorno de Anaconda para asegurarse de que tiene exactamente lo que necesita para ese proyecto. Puede usar el método o conjunto de herramientas que prefiera. La mayoría de estos módulos no requieren explícitamente Visual Studio Code ni Azure.

### Configuración del entorno local

Antes de continuar, asegúrese de que tiene lo siguiente:

* Visual Studio Code, Anaconda y Python instalados. Crearemos nuestro entorno de Anaconda en los pasos siguientes.
* Una carpeta local que haya creado para almacenar todo el código y los datos.
* El archivo de Excel de nuestros datos descargado y guardado en su carpeta local.
* Un cuaderno de Jupyter Notebook en blanco guardado en la carpeta. En la carpeta local, cree un archivo ficticio llamado *nombredearchivo.ipynb*.

Para configurar el entorno local:

1. Abra el símbolo del sistema de Anaconda.

    ![anaconda-prompt](https://learn.microsoft.com/es-es/training/modules/collect-manipulate-data-python-nasa/media/anaconda-prompt.png)

2. En el símbolo del sistema de Anaconda, cree un nuevo entorno de Anaconda con Pandas, NumPy, scikit-learn, PyDotPlus y Jupyter:

    ```
    conda create -n myenv python=3.8 pandas numpy jupyter seaborn scikit-learn pydotplus
    ```

3. En el símbolo del sistema de Anaconda, active el nuevo entorno:

    ```
    conda activate myenv
    ```

4. En el símbolo del sistema de Anaconda, instale AzureML-SDK:

    ```
    pip install --upgrade azureml-sdk
    ```

    En algunos casos, la instalación puede tardar varios minutos en completarse. Deje que se resuelva hasta que se complete.

5. En el símbolo del sistema de Anaconda, instale un lector de Excel (tenga en cuenta que xlrd podría no funcionar con el archivo de datos de Excel que descargó):

    ```
    pip install openpyxl
    ```

6. En Visual Studio Code, abra la carpeta local que ha creado para almacenar todo el código y los datos. Seleccione la versión de Python del kernel de Jupyter de la parte superior derecha y el intérprete de Python de la parte inferior izquierda, y establézcalos para usar su entorno de Anaconda:

    ![ensure-python](https://learn.microsoft.com/es-es/training/modules/collect-manipulate-data-python-nasa/media/ensure-python.png)

### Importación de bibliotecas

Con el entorno local de Visual Studio Code creado, ahora puede importar las bibliotecas. Le ayudarán a importar y limpiar los datos meteorológicos, y a crear y probar el modelo de Machine Learning.

Copie el código siguiente en una celda y ejecútelo para importar las bibliotecas.

```python
# Pandas library is used for handling tabular data
import pandas as pd

# NumPy is used for handling numerical series operations (addition, multiplication, and ...)
import numpy as np

# Sklearn library contains all the machine learning packages we need to digest and extract patterns from the data
from sklearn import linear_model, model_selection, metrics
from sklearn.model_selection import train_test_split

# Machine learning libraries used to build a decision tree
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree

# Sklearn's preprocessing library is used for processing and cleaning the data 
from sklearn import preprocessing

# for visualizing the tree
import pydotplus
from IPython.display import Image
```

### Lectura de datos en una variable

Ahora que se han importado todas las bibliotecas, puede usar la biblioteca pandas para importar los datos. Use el comando `pd.read_excel` para leer los datos y guardarlos en una variable. Después, use la función `.head()` para imprimir las primeras cinco filas de datos para asegurarse de que lo hemos leído todo correctamente.

```python
launch_data = pd.read_excel('RocketLaunchDataCompleted.xlsx')
launch_data.head()
```

### Inicio de la exploración de los datos

Por último, podemos usar la llamada de función `.columns` para ver todas las columnas de nuestros datos. Así, se nos mostrarán los atributos que tienen los datos. Verá algunos atributos comunes, como los nombres de los cohetes anteriores programados para el lanzamiento, las fechas en que se han programado, si realmente se han lanzado y muchos más. Examine estas columnas e intente adivinar cuáles tendrán el mayor impacto a la hora de determinar si se va a lanzar un cohete.

```python
launch_data.columns
```

<hr/>

## Ejercicio: Limpieza de los datos meteorológicos para analizar los criterios de lanzamiento de cohetes

Ahora que se han importado los datos, es necesario aplicar una práctica de aprendizaje automático conocida como "limpieza de los datos". Tomamos datos que parecen incorrectos o desordenados y los limpiamos cambiando el valor o eliminándolos por completo. Ejemplos comunes de limpieza de datos:

* Asegurarse de que no hay valores "null".
* Hacer que todos los valores de una columna sean iguales.
* Limpiamos datos porque los equipos se confunden si ven datos incoherentes o si muchos valores en los datos son "null".

### Limpieza de datos

El primer paso para limpiar los datos consiste en reemplazar por algo todos los valores que faltan. Normalmente, para sustituir estos valores se requiere experiencia en la materia. Sin embargo, en este caso, seguirá su mejor criterio. En algunas filas (recuerde que representan días) faltan datos sobre el tiempo o los lanzamientos.

Para empezar, obtenga una visión general de los datos sobre lanzamientos ejecutando este comando en su archivo *.ipynb:*

```python
launch_data.info()
```

De 300 filas, en algunas columnas falta información:

```
RangeIndex: 300 entries, 0 to 299
Data columns (total 26 columns):
 #   Column                        Non-Null Count  Dtype         
---  ------                        --------------  -----         
 0   Name                          60 non-null     object        
 1   Date                          300 non-null    datetime64[ns]
 2   Time (East Coast)             59 non-null     object        
 3   Location                      300 non-null    object        
 4   Crewed or Uncrewed            60 non-null     object        
 5   Launched?                     60 non-null     object        
 6   High Temp                     299 non-null    float64       
 7   Low Temp                      299 non-null    float64       
 8   Ave Temp                      299 non-null    float64       
 9   Temp at Launch Time           59 non-null     float64       
 10  Hist High Temp                299 non-null    float64       
 11  Hist Low Temp                 299 non-null    float64       
 12  Hist Ave Temp                 299 non-null    float64       
 13  Precipitation at Launch Time  299 non-null    float64       
 14  Hist Ave Precipitation        299 non-null    float64       
 15  Wind Direction                299 non-null    object        
 16  Max Wind Speed                299 non-null    float64       
 17  Visibility                    299 non-null    float64       
 18  Wind Speed at Launch Time     59 non-null     float64       
 19  Hist Ave Max Wind Speed       0 non-null      float64       
 20  Hist Ave Visibility           0 non-null      float64       
 21  Sea Level Pressure            299 non-null    object        
 22  Hist Ave Sea Level Pressure   0 non-null      float64       
 23  Day Length                    298 non-null    object        
 24  Condition                     298 non-null    object        
 25  Notes                         3 non-null      object
```

Puede ver que en `Hist Ave Max Wind Speed`, `Hist Ave Visibility` y `Hist Ave Sea Level Pressure` no hay datos.

Parece lógico que `Wind Speed at Launch Time`, `Temp at Launch Time`, `Launched`, `Crewed or Uncrewed`, `Time` y `Name` solo tengan 60 valores, ya que los datos solo incluyen 60 lanzamientos. El resto son los días antes y después del lanzamiento.

Estas son algunas formas de limpiar los datos:

* En las filas que no tienen `Y` en la columna `Launched` no había lanzamientos de cohete, por lo que esos valores que faltan se convierten en `N`.
* En el caso de las filas en las que falta información sobre si el cohete estaba tripulado o no, se asume que no lo estaba. Es más probable que fuera sin tripulación porque hubo menos misiones tripuladas.
* Si falta la dirección del viento, márquelo como `unknown`.
* Si faltan datos sobre las condiciones, imagine que se trata de un día típico y use `fair`.
* Para cualquier otro dato, use un valor de `0`.

En la celda siguiente, pegue y ejecute este código:

```python
## To handle missing values, we will fill the missing values with appropriate values 
launch_data['Launched?'].fillna('N',inplace=True)
launch_data['Crewed or Uncrewed'].fillna('Uncrewed',inplace=True)
launch_data['Wind Direction'].fillna('unknown',inplace=True)
launch_data['Condition'].fillna('Fair',inplace=True)
launch_data.fillna(0,inplace=True)
launch_data.head()
```

Intente ejecutar de nuevo `launch_data.info()` para ver los cambios que acaba de realizar en los datos.

### Manipulación de datos

Como los cálculos son más adecuados para entradas numéricas, convierta todo el texto en números. Por ejemplo, se usará `1` si un cohete está tripulado y `0` si no lo está.

```python
## As part of the data cleaning process, we have to convert text data to numerical because computers understand only numbers
label_encoder = preprocessing.LabelEncoder()

# Three columns have categorical text info, and we convert them to numbers
launch_data['Crewed or Uncrewed'] = label_encoder.fit_transform(launch_data['Crewed or Uncrewed'])
launch_data['Wind Direction'] = label_encoder.fit_transform(launch_data['Wind Direction'])
launch_data['Condition'] = label_encoder.fit_transform(launch_data['Condition'])
```

Ahora, vuelva a examinar todos los datos y compruebe que se han limpiado.

```python
launch_data.head()
```

<hr/>

## Ejercicio: Datos adicionales que se podrían incluir

Las decisiones realizadas en este módulo eran simplistas, en el mejor de los casos. Aunque el día anterior o posterior al lanzamiento de la SpaceX Dragon el 30 de mayo de 2020 no se lanzó ningún cohete, no significa que se haya aplazado un lanzamiento debido a las condiciones meteorológicas de esos días. Por eso, es impreciso colocar `N` en la columna `Launched` de esas fechas.

En estos módulos se le guía por los pasos prácticos que se han seguido para solucionar los problemas que se han tenido que abordar durante la exploración espacial. Pero también se espera que descubra su propia ruta de acceso. El objetivo final es que consiga la inspiración para crear, imaginar y forzar los límites de lo que sabemos y del conocimiento que tenemos de este mundo y el más allá.

Estas son algunas formas de continuar con el aprendizaje y el recorrido por los datos:

* **Explore los datos con más detalle**: busque artículos e informes sobre cada lanzamiento. ¿Se realizaron consideraciones sobre el tiempo antes del lanzamiento? ¿Las condiciones meteorológicas en torno a estas fechas se podrían haber considerado preocupantes?
* **Explore los datos meteorológicos que faltan**: ¿qué ocurre con las fechas en las que la NASA decidió no lanzar cohetes? Más allá de días concretos, ¿la NASA ha evitado alguna estación? ¿Qué tipo de perfil meteorológico suelen tener esas estaciones?
* **Explore los datos de lanzamiento que faltan**: ¿puede encontrar datos sobre los lanzamientos que se aplazaron debido a las condiciones meteorológicas? ¿Hay datos sobre lanzamientos de otros países o regiones que pueda incorporar?
* **Explore otras manipulaciones de datos**: ¿Se podrían haber usado mejores valores para rellenar los datos que faltaban?
* **Decida qué datos le gustaría tener**: si tuviera acceso a los expertos en la materia y los orígenes de datos de la NASA, ¿qué cree que sería más importante para tomar una decisión sobre un lanzamiento o un aplazamiento? Si pudiera preguntarle a un experto, ¿qué le preguntaría?
* **Evalúe problemas similares**: ¿hay problemas similares que puede usar para rellenar estos datos? Por ejemplo, ¿los retrasos de vuelos en la zona debido a las condiciones meteorológicas también son un indicador?

Ser científico de datos no consiste en tener un conjunto de datos completo y aplicarle un algoritmo. Se trata de empezar desde un punto de partida desordenado y probablemente incorrecto, y aprender de manera iterativa. Al final de la ruta de aprendizaje, habrá aprendido algo sobre el lanzamiento de cohetes. Con ese nuevo conocimiento, puede volver a este módulo y tomar decisiones mejor fundamentadas.