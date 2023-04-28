# Compilación de un modelo de Machine Learning 💻🤖

En este módulo, se centrará en un análisis local de los datos mediante scikit-learn y usará un clasificador de árbol de decisión para obtener información de datos meteorológicos y de lanzamiento de cohetes sin procesar.

### Objetivos de aprendizaje:

* La importancia de la elección de las columnas.
* El procedimiento para dividir los datos a fin de entrenar y probar eficazmente un algoritmo de aprendizaje automático.
* El procedimiento para entrenar, probar y puntuar un algoritmo de aprendizaje automático.
* El procedimiento para visualizar un modelo de clasificación de árbol.

<hr/>

## Introducción

Antes, ha importado 300 filas de datos meteorológicos que representaban 60 lanzamientos de cohetes, más los dos días anteriores y posteriores a un lanzamiento. A través de una versión simplista de limpieza y manipulación de datos, ha llevado los datos a un lugar donde puede empezar a usar algoritmos de aprendizaje automático para recopilar información sobre ellos.

En este módulo, usará un clasificador de árbol de decisión para obtener información de datos meteorológicos y de lanzamiento de cohetes sin procesar. Este módulo se centrará en un análisis local de los datos mediante scikit-learn.

<hr/>

## Ejercicio: Determinación de las columnas que se van a incluir en un modelo de Machine Learning

Para empezar el entrenamiento del modelo de Machine Learning, hay que enseñar al equipo qué partes de los datos examinar para realizar predicciones. Ya sabe que "Launched" es la columna que quiere que prediga el modelo. Extraerá esta columna y la almacenará en una variable como una lista de `Y` y `N`.

### Limpieza adicional de los datos

A continuación, se quitarán algunas de las columnas que no son necesarias para realizar esta predicción. Las columnas como "Name" proporcionan más contexto sobre los datos. Sin embargo, el nombre de un lanzamiento no es un indicador de si las condiciones meteorológicas harán que se posponga. En este módulo, se centrará en las columnas correspondientes a la velocidad del viento, las condiciones meteorológicas y las precipitaciones.

En el cuaderno de Jupyter Notebook (archivo *.ipynb*) que creó en el módulo anterior, ejecute los siguientes comandos. Si ha transcurrido demasiado tiempo desde que ejecutó los pasos de ese módulo, es posible que observe errores. Si este es el caso, vuelva a importar las bibliotecas y los datos del módulo anterior y, luego, ejecute los comandos:

```python
# First, we save the output we are interested in. In this case, "launch" yes and no's go into the output variable.
y = launch_data['Launched?']

# Removing the columns we are not interested in
launch_data.drop(['Name','Date','Time (East Coast)','Location','Launched?','Hist Ave Sea Level Pressure','Sea Level Pressure','Day Length','Notes','Hist Ave Visibility', 'Hist Ave Max Wind Speed'],axis=1, inplace=True)

# Saving the rest of the data as input data
X = launch_data
```

Ahora tiene dos variables. La salida está en `y` y la entrada en `X`. Puede ver una introducción de los datos de entrada si examina las columnas de la variable `X` recién creada:

```python
# List of variables that our machine learning algorithm is going to look at:
X.columns
```

Los datos de entrada `X` representan el tiempo de un día concreto. En este caso, la fecha o la hora son irrelevantes. Queremos que el perfil de las condiciones meteorológicas para ese día, y no la fecha o la hora, sea el indicador de si se debe producir un lanzamiento.

<hr/>

## Ejercicio: Elección del algoritmo de aprendizaje automático para predecir el éxito del lanzamiento de un cohete

Ha elegido las columnas que quiere usar para predecir si, bajo determinadas condiciones meteorológicas, se puede lanzar un cohete o no. Ahora, tendrá que elegir el algoritmo que se va a usar para crear el modelo. Recuerde que ya ha visto la [hoja de referencia rápida de algoritmos de Azure Machine Learning](https://learn.microsoft.com/es-es/azure/machine-learning/algorithm-cheat-sheet).

![algorithm-cheat-sheet](https://learn.microsoft.com/es-es/training/modules/machine-learning-model-nasa/media/algorithm-cheat-sheet.png)

Recuerde la pregunta: *¿Se puede predecir si existe la probabilidad de que se produzca un lanzamiento bajo condiciones meteorológicas específicas?* Esta pregunta tiene dos opciones. Un cohete se lanzará, sí o no. Esta pregunta se considera un problema de **clasificación de dos clases**.

Dentro de esta categoría de algoritmos, hay muchos específicos entre los que elegir. En este caso, va a explorar un **árbol de decisión de dos clases**. La visualización de los resultados de un árbol de decisión le proporciona ideas que le ayudarán a realizar la iteración en la recopilación, la limpieza y la manipulación de los datos en el futuro.

### Creación de un modelo de Machine Learning en Python

Con [scikit-learn](https://scikit-learn.org/stable/index.html), resulta fácil crear el modelo de Machine Learning que necesita para este ejercicio. Pegue este código en otra celda de Visual Studio Code:

```python
# Create decision tree classifier 
tree_model = DecisionTreeClassifier(random_state=0,max_depth=5)
```

Echemos un vistazo a la [documentación del clasificador de árbol de decisión](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html?highlight=decision%20tree%20classifier#sklearn.tree.DecisionTreeClassifier?azure-portal=true). Comprenda la importancia de los dos parámetros especificados aquí: `random_state` y `max_depth`.

El parámetro `random_state` se usa con la mayoría de los algoritmos de aprendizaje automático. Controla la aleatoriedad del algoritmo. Cuando se usa este estimador para dividir los datos en datos de entrenamiento y datos de prueba, la inicialización proporcionada aquí determina la aleatoriedad de esa división. En la unidad siguiente se proporciona más información sobre la división de los datos.

El parámetro `max_depth` es un parámetro específico del árbol que le permite limitar el ámbito de la salida del modelo. En este caso, no es tan informativo conocer cada una de las probabilidades posibles de una condición meteorológica específica y cómo podría afectar a la probabilidad del lanzamiento de un cohete. La profundidad se limita a cinco para reducir el conocimiento adquirido a lo que está prácticamente más relacionado con el resultado.

### Exploración adicional

Si le interesa, puede intentar finalizar este módulo tal cual. Después, puede volver y cambiar los valores de los parámetros para ver qué tipo de conclusiones nuevas podría obtener.

<hr/>

## Ejercicio: División de los datos en conjuntos de entrenamiento y de prueba

El siguiente paso consiste en dividir los datos en conjuntos de entrenamiento y de prueba. Proporcionar todos los datos al clasificador de aprendizaje automático solo servirá para que le indique qué datos tiene. No devolverá predicciones precisas.

### ¿Por qué debe dividir los datos?

Una manera de explicar la importancia de la división de los datos es compararla con un examen que podría hacer en una clase de educación oficial. Durante la clase, se le muestran problemas de ejemplo y se le dicen las respuestas. Este escenario se produce en conferencias, deberes y exámenes prácticos.

Imagine una clase en la que el profesor le facilita el examen exacto y las respuestas el día antes. ¿Obtendrá un examen perfecto? Sí.

¿Sabría si ha aprendido los conceptos? No. Es más probable que haya aprendido las respuestas a las preguntas del examen que no los conceptos que el examen intentaba probar.

Si *realmente* quiere aprender, debe practicar con problemas para los que tiene las respuestas. Cuando se sienta seguro de estos problemas, intente solucionar otros para los que todavía no conoce las respuestas. Así es básicamente cómo "aprende" el clasificador de aprendizaje automático.

### División de los datos

Quiere dividir los datos en cuatro variables nuevas. Ya tiene `X` e `y`, que representan la entrada y la salida. Ahora es el momento de dividirlas en datos de entrenamiento y de prueba.

Mediante scikit-learn y la función de división del clasificador, puede obtener un muestreo aleatorio de `X` e `y` que coincidan en orden. Si no dividió los datos aleatoriamente, sino que tomó el primer 80% de las filas de los datos de entrenamiento y dejó el resto para las pruebas, se producirán problemas.

Por ejemplo, imagine que los datos están ordenados por fecha. Si se tomaran las primeras 240 filas para el entrenamiento, tendría que entrenar el modelo con datos anteriores a 1999. Este escenario supone un problema porque es posible que los sensores hayan cambiado con el tiempo. El examen *exclusivo* de datos más antiguos podría no ser un buen indicador para las decisiones que se realizarían con la nueva tecnología.

Pegue este código en una celda de Visual Studio Code para dividir los datos:

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=99)
```

Este código separa de forma aleatoria los datos en cuatro grupos: X_train, X_test, y_train e y_test. Con la función train_test_split de scikit-learn, se especifican cuatro parámetros importantes:

* Datos de entrada: `X`; todas las columnas que se quieren usar para predecir un lanzamiento.
* Datos de salida: `y`; el resultado de cada fila (si un cohete se ha lanzado o no).
* Tamaño de la prueba: `0.2`; el entrenamiento con el 80% de los datos y las pruebas con el 20% restante es una división habitual en ciencia de datos.
* Estado aleatorio: `99`; un valor de inicialización aleatorio que cambiará la aleatoriedad de la selección de los datos.

También puede volver y modificar el tamaño de la prueba y el estado aleatorio para probar otras opciones.

<hr/>

## Ejercicio: Entrenamiento y prueba del modelo de Machine Learning para predecir el éxito del lanzamiento de un cohete

Una vez que los datos se han separado en las secciones de entrenamiento y prueba, se puede entrenar el modelo de Machine Learning. Uno de los motivos de la popularidad del lenguaje Python para la ciencia de datos y el aprendizaje automático son todas las bibliotecas que existen para admitir el estudio de los datos. Como se ha visto, la creación del modelo de aprendizaje automático y la división de los datos han sido sencillas. El ajuste y la prueba del modelo también lo serán.

### Ajuste del modelo

El siguiente paso del ciclo de vida de la ciencia de datos consiste en ajustar el modelo a los datos de entrenamiento. La acción de "ajuste" es básicamente la manera en la que el modelo aprende. Este proceso se describe con el ejemplo de las bayas. El "ajuste" se producía cuando la persona traía una baya y se le decía de qué tipo era. Para ajustar el modelo, debe llamar a `fit()` en el clasificador de aprendizaje automático y pasar los datos de `X_train` e `y_train`.

El ajuste del modelo es como realizar un examen práctico en el que tiene acceso a las respuestas para asegurarse de que comprende los conceptos.

```python
# Fitting the model to the training data
tree_model.fit(X_train, y_train)
```

### Prueba del modelo

La prueba del modelo también se ha facilitado con las bibliotecas que se han importado. La prueba del modelo es como realizar el examen. Proporcionará `X_test` (el 20% de los datos de entrada que ha reservado para la prueba) a la función `predict()` del clasificador. Esta función devuelve una lista de `Y` y `N` que representa lo que el modelo cree que ocurrirá si se intentara lanzar un cohete dado un conjunto determinado de condiciones meteorológicas.

Pegue el código siguiente en Visual Studio Code para la predicción y, después, imprima las predicciones.

```python
# Do prediction on test Data
y_pred = tree_model.predict(X_test)
print(y_pred)
```

¿Cuántos valores `Y` ha obtenido? ¿Las predicciones parecen representativas de los datos que se han introducido? No resulta claro sin una mayor investigación, pero hasta ahora la salida contiene aproximadamente 9 respuestas `Y` de los 60 valores de entrada. Aproximadamente el 20% del total de datos han generado un resultado `Y`. Nuestro porcentaje es aproximadamente el 15% de estos datos pronosticados, por lo que se acerca bastante.

<hr/>

## Ejercicio: Puntuación del modelo de Machine Learning de predicción del éxito del lanzamiento de un cohete

Realizar una comparación simple del porcentaje de datos que han devuelto `Y` para el lanzamiento es útil para comprobar si el modelo está cerca de ser correcto. Pero puntuar realmente el modelo es incluso más útil.

### Puntuación del modelo

Como sucede con un examen, el aprendizaje se puede medir con una puntuación. Hay una función de una línea a la que se puede llamar para ver el grado de exactitud del modelo con respecto a la probabilidad de que se realice un lanzamiento.

```python
# Calculate accuracy
tree_model.score(X_test, y_test)
```

Con la función `score()`, se pasan los datos de entrada `X_test` y los datos de salida `y_test` para "evaluar" el modelo. Cuanto mayor sea la puntuación, mejor será el modelo en la predicción del resultado del lanzamiento de un cohete en función de los datos meteorológicos.

### Descripción de la puntuación

El modelo de este ejemplo tiene una exactitud del 98,3%, que es un buen valor. De hecho, con la pequeña cantidad de limpieza y manipulación de datos que se ha realizado y con los problemas conocidos de los datos, parece demasiado bueno.

Es posible que tanta precisión se deba a que los datos son los mejores y a que el modelo se ha entrenado correctamente. Pero también puede ser por la facilidad para adivinar estos datos fabricados parcialmente. Por lo tanto, esta puntuación no sería confiable en el mundo real. Para contextualizar, una precisión del 70% con un clasificador de árbol de decisión (la primera vez que se ejecuta) es habitual.

¿Cómo podría garantizar que la puntuación es una representación exacta de la precisión del modelo?

Una manera sería pedirle a un experto que rellene los datos `Y` y `N` de las fechas en las que no hay lanzamientos en lugar de simplemente adivinar `N`. Por ejemplo, la probabilidad de que el día inmediatamente anterior o posterior a un lanzamiento *también* sea un buen día para un lanzamiento seguramente sea mayor de lo que se ha representado en estos datos.

<hr/>

## Ejercicio: Visualización del modelo de Machine Learning

Una de las ventajas de utilizar un clasificador de árbol de decisión es la visualización que puede usar para comprender mejor cómo el modelo toma las decisiones. Con `graphviz` y `pydotplus`, puede ver rápidamente cómo se toma una decisión. En futuras iteraciones, puede ver cómo se cambian las decisiones.

### Creación del árbol visual

Para crear una representación visual del modelo, debe crear una función que tome los siguientes elementos como parámetros:

* Datos: `tree`; el modelo de Machine Learning
* Columnas: `feature_names`; una lista de las columnas de los datos de entrada
* Salida: `class_names`; una lista de las opciones de clasificación (en este caso, Sí o No)
* Nombre de archivo: `png_file_to_save`; el nombre del archivo en el que quiere guardar la visualización

Llamará a la función export_graphviz() de scikit-learn y, después, devolverá una imagen que representa el grafo generado de forma automática por scikit-learn.

```python
# Let's import a library for visualizing our decision tree.
from sklearn.tree import export_graphviz

def tree_graph_to_png(tree, feature_names, class_names, png_file_to_save):
    tree_str = export_graphviz(tree, feature_names=feature_names, class_names=class_names,
                                     filled=True, out_file=None)
    graph = pydotplus.graph_from_dot_data(tree_str)  
    return Image(graph.create_png())
```

La llamada de esta función es muy sencilla:

* Datos: `tree_model`; el modelo que ha entrenado y probado antes
* Columnas: `X.columns.values`; la lista de columnas de la entrada
* Salida: [`yes`, `no`]; los dos resultados posibles
* Nombre de archivo: `decision_tree.png`; el nombre del archivo en el que quiere guardar la imagen

```python
# This function takes a machine learning model and visualizes it.
tree_graph_to_png(tree=tree_model, feature_names=X.columns.values,class_names=['No Launch','Launch'], png_file_to_save='decision-tree.png')
```

![decision-tree](https://learn.microsoft.com/es-es/training/modules/machine-learning-model-nasa/media/decision-tree.png)

En general, cuando se examina el conjunto de datos, hay 240 muestras:

* 192 son lanzamientos cancelados
* 48 son lanzamientos

Este resultado se debe a la estrategia de limpieza de datos, donde se ha asumido que todos los días sin etiquetar son días sin lanzamientos.

Con las etiquetas nuevas, se puede decir "Si la velocidad del viento era inferior a 1,0, 191 de las 240 muestras han adivinado que en ese día no fue posible ningún lanzamiento". Este resultado podría parecer extraño, pero es correcto según los datos. Aquí tenemos la prueba: se ha trazado la distribución entre los días con y sin lanzamientos en los que la velocidad del viento en el momento del lanzamiento < = 1 antes de quitar la columna anteriormente en este cuaderno. se muestra que casi ninguna vez se produce el lanzamiento:

![plot-launches](https://learn.microsoft.com/es-es/training/modules/machine-learning-model-nasa/media/plot-launches.png)

### Descripción de la visualización

En este árbol simple se muestra que la característica más importante de los datos ha sido `Wind Speed at Launch Time`. Si la velocidad del viento era inferior a 1,0, en 191 de las 240 muestras se ha adivinado correctamente que no se realizaría el lanzamiento. Se aprecia que solo 191 de esas muestras necesitaban que el valor de `Wind Speed at Launch Time` fuera inferior a 1,0 para adivinar correctamente el resultado, mientras que por encima de 1,0 se necesita más información.

Estas conclusiones no son correctas. Anteriormente todos los valores que estaban vacíos se han establecido en 0. También se sabe que muchos de los valores que estaban relacionados con el tiempo de inicio eran 0 porque el 60% de los datos no estaban relacionados con un lanzamiento real o un intento de lanzamiento.

Si continúa el examen del árbol, puede ver que `Max Wind Speed` es la siguiente característica más importante de los datos. En este caso, se aprecia que, de los 49 días restantes en los que la velocidad máxima del viento era inferior a 30,5, en 48 de ellos se devolvió una salida de lanzamiento correcto y en uno una salida sin lanzamiento.

Estos datos podrían ser más interesantes con un contexto del mundo real. Solo hubo un día en el que se había planificado un lanzamiento y el valor de `Max Wind Speed` era mayor que 30,5: el 27 de mayo de 2020. El lanzamiento de la Space X Dragon se pospuso al 30 de mayo de 2020. Estas son las pruebas:

```python
launch_data[(launch_data['Wind Speed at Launch Time'] > 1) & (launch_data['Max Wind Speed'] > 30.5)]
```

### Mejora de los resultados

Con esta visualización, podría ver que algunas características han pasado a ser importantes. Pero este énfasis se ha basado en información incorrecta.

Una mejora que se podría hacer consiste en determinar la relación entre `Max Wind Speed` y `Wind Speed at Launch Time` para las filas que tienen esa información. Después, en lugar de establecer `Wind Speed at Launch Time` en 0 para los días en los que no hay lanzamientos, se podría haber convertido en la estimación de lo que sería una hora de lanzamiento común. Este cambio podría haber representado mejor los datos.

¿Se le ocurren otras formas de mejorar los datos?

<hr/>

## Ejercicio: Predicción del éxito del lanzamiento de un cohete mediante el aprendizaje automático

Por último, es el momento de probar el modelo con datos que nunca han estado en el conjunto de datos.

El 30 de julio de 2020, la NASA lanzó el vehículo Perseverance a Marte desde Cabo Cañaveral a las 7:50, hora oriental.

Recopile los datos de entrada para el modelo:

* Con o sin tripulación
* Temperatura máxima
* Temperatura mínima
* Temperatura media
* Temperatura en el momento del lanzamiento
* Historial de temperaturas máximas
* Historial de temperaturas mínimas
* Historial de temperaturas medias
* Precipitación en el momento del lanzamiento
* Historial de precipitación media
* Dirección del viento
* Velocidad máxima del viento
* Visibilidad
* Velocidad del viento en el momento del lanzamiento
* Historial de la velocidad máxima del viento
* Historial de visibilidad media
* Condición

Puede encontrar esta información en la mayoría de los sitios meteorológicos. Recuerde que todos los datos deben ser numéricos.

En el ejemplo siguiente se usan datos hipotéticos:

```python
# ['Crewed or Uncrewed', 'High Temp', 'Low Temp', 'Ave Temp',
#        'Temp at Launch Time', 'Hist High Temp', 'Hist Low Temp',
#        'Hist Ave Temp', 'Precipitation at Launch Time',
#        'Hist Ave Precipitation', 'Wind Direction', 'Max Wind Speed',
#        'Visibility', 'Wind Speed at Launch Time', 'Hist Ave Max Wind Speed',
#        'Hist Ave Visibility', 'Condition']

data_input = [ 1.  , 75.  , 68.  , 71.  ,  0.  , 75.  , 55.  , 65.  ,  0.  , 0.08,  0.  , 16.  , 15.  ,  0.  ,  0. ]

tree_model.predict([data_input])
```

### Continuidad de la mejora

A medida que siga mejorando el modelo como se describe en esta ruta de aprendizaje, puede consultar otros [lanzamientos de cohetes de la NASA](https://www.nasa.gov/launchschedule/). Vea si el modelo puede predecir con precisión los resultados.

También puede usar las predicciones meteorológicas combinadas con el modelo de Machine Learning para ver si puede predecir un retraso incluso antes de que se realicen los lanzamientos.