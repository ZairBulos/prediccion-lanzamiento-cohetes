# Introducción al lanzamiento de cohetes 🚀🔥

Obtenga una introducción sobre la forma en que la NASA elige una fecha para el lanzamiento de un cohete y descubra algunos aspectos básicos del aprendizaje automático.

### Objetivos de aprendizaje

* Los desafíos que la meteorología puede suponer para el lanzamiento de un cohete
* El ciclo de vida de la ciencia de datos
* El funcionamiento del aprendizaje automático
* El papel de la ética en el aprendizaje automático

<hr/> 

## Introducción

La Administración Nacional de Aeronáutica y el Espacio (NASA) recopila datos de muchos orígenes diferentes para ayudar a predecir si las condiciones serán seguras para el lanzamiento previsto de un cohete. En el caso de los lanzamientos tripulados y no tripulados, es crucial que las condiciones meteorológicas no afecten a la trayectoria y la velocidad de la nave en su recorrido hacia el espacio.

La NASA tiene en cuenta muchos factores potenciales cuando selecciona un lugar para el lanzamiento de cohetes. La agencia espacial debe tener en cuenta muchos orígenes de datos para obtener la imagen más precisa de las condiciones a las que se debe enfrentar el cohete en su viaje de 100 kilómetros hacia el espacio.

La NASA ha descubierto que es más seguro que un cohete lanzado desde la Tierra se adentre en el espacio en unas condiciones atmosféricas específicas. Además, para aumentar las posibilidades de éxito, los cohetes deben entrar en órbita lo más cerca posible a la línea del Ecuador. Si es necesario abortar un lanzamiento, es más seguro hacerlo sobre una gran masa de agua. Cabo Cañaveral, en Florida, es un lugar que cumple todos estos requisitos.

![cape-canaveral](https://learn.microsoft.com/es-es/training/modules/introduction-rocket-launch-nasa/media/cape-canaveral.png)

Cabo Cañaveral se encuentra en la costa este de la península de Florida, en un área en la que suelen registrarse factores ambientales que podrían afectar al lanzamiento de un cohete, como humedad alta y frecuentes huracanes, tornados y temporales de lluvias. Entonces, ¿por qué no se hace el lanzamiento desde un lugar sin estas inclemencias meteorológicas, como San Diego, en California?

Si se tienen en cuenta todos los factores, a pesar de la meteorología, Cabo Cañaveral se considera el mejor lugar de lanzamiento de los Estados Unidos para garantizar que se mantenga la trayectoria necesaria para que un cohete llegue al espacio.

<hr/>

## Datos para predecir las condiciones meteorológicas con años de antelación

Se tarda años en desarrollar los planes para llevar a cabo con éxito el lanzamiento seguro de un cohete. Debido a esto, la NASA podría tener que elegir una fecha y una hora varios años antes del lanzamiento real del cohete.

Como se dispone de grandes cantidades de datos en el mundo digital, las predicciones meteorológicas son más precisas que en el pasado, e incluso se tienen en cuenta las condiciones cambiantes del clima. Pero si se fija en las previsiones meteorológicas que realiza el meteorólogo local, se dará cuenta de que puede ser todo un desafío predecir con precisión el tiempo aunque solo sea con unas horas de antelación. El riesgo durante el lanzamiento de un cohete es muy alto. Si la NASA programa el lanzamiento para un día en el que se acaba produciendo un evento meteorológico, la decisión de programarlo ese día podría poner vidas en riesgo.

Para facilitar la predicción y el análisis de los datos meteorológicos, la NASA colabora de forma estrecha con la [Oficina Nacional de Administración Oceánica y Atmosférica (NOAA)](https://www.noaa.gov/). La NOAA comparte sus datos con el público, por lo que puede empezar a analizar los patrones y realizar las mismas predicciones que la NASA. Puede acceder a los datos de la NOAA de varias maneras. En la sección [Acceso a datos de la NOAA](https://www.ncei.noaa.gov/), encontrará varias [API](https://en.wikipedia.org/wiki/API) que proporcionan acceso a los datos de la NOAA a través del código. También puede descargar los datos sin costo alguno, o incluso solicitar copias impresas de los datos a un precio reducido.

La NASA también recopila sus propios datos, que pone a disposición del público. En [Data.NASA.gov](https://nasa.github.io/data-nasa-gov-frontpage/) puede encontrar decenas de miles de conjuntos de datos. La NASA también proporciona [recursos para desarrolladores](https://data.nasa.gov/stories/s/gk8h-th3y) destinados a todos aquellos que quieran integrar los datos de la NASA en sus aplicaciones.

Los expertos de la NASA usan todos estos datos para garantizar la mejor predicción posible y seleccionar una fecha de lanzamiento segura.

<hr/>

## Análisis meteorológico del día del lanzamiento

Aunque los datos históricos y de predicción son esenciales, la NASA debe analizar muchos factores críticos con vistas al día del lanzamiento de un cohete.

La NASA recopila datos de una amplia gama de orígenes, entre los que se incluyen:

* Globos aerostáticos a gran altitud
* Predicciones meteorológicas de la NOAA
* Imágenes por satélite
* Sensores remotos
* Expertos en patrones meteorológicos

Estos son algunos factores clave que la NASA tiene en cuenta para preparar un lanzamiento:

* [Campo eléctrico de la superficie](https://www.nasa.gov/scientificballoons/)
* [Hora](https://www.noaa.gov/weather)
* [Distancia]()
* [Temperatura de las nubes](https://weather.ndc.nasa.gov/GOES/)
* [Intensidad de las precipitaciones](https://www.earthdata.nasa.gov/sensors)
* Velocidad del cohete

La división 45th Space Wing de la Fuerza Espacial de los Estados Unidos se centra en la exploración espacial y la meteorología relacionada con los lanzamientos de cohetes exitosos. Esta división ha creado un [gráfico sencillo](https://www.patrick.spaceforce.mil/Portals/14/Weather/LaunchFAQ.pdf?ver=wW9MREd4NYgnoOMcKgGG2A%3d%3d) en el que se muestran algunos de los criterios fundamentales que se deben tener en cuenta para un lanzamiento seguro.

### Información que se puede obtener de las nubes

uede obtener mucha información sobre la meteorología local si observa las nubes. La forma, la composición y el movimiento de las nubes pueden ayudarle a predecir el tiempo que hará, incluida la temperatura o si caerán rayos.

La temperatura y los rayos son factores meteorológicos fundamentales para el lanzamiento de un cohete. Cada cohete tiene diferentes requisitos de temperatura, en función de cómo se haya construido y de su carga. En función de estos factores, cada cohete tiene su propio umbral de temperatura, y esta información es exclusiva del cohete. Pero por lo general, hay umbrales de temperatura que determinan el éxito o el fracaso de un lanzamiento.

Como ya se imaginará, también es fundamental que no caiga ningún rayo durante el lanzamiento. Como los cohetes tienen una alta conductividad, los posibles daños de los rayos son extremadamente elevados y peligrosos. Bajo determinadas condiciones, el lanzamiento de un cohete puede incluso generar rayos. La forma y la composición de las nubes son indicadores de la probabilidad de que haya rayos.

La NASA analiza todos estos factores meteorológicos cuando decide si se realiza el lanzamiento de un cohete y cuál será su trayectoria. Se pueden obtener indicadores e información esencial si se observan las nubes.

<hr/>

## El aprendizaje automático y el ciclo de vida de la ciencia de datos

El aprendizaje automático forma parte de un campo más amplio: la ciencia de datos. Consiste esencialmente en el proceso de creación de conocimiento a partir de datos sin procesar.

Se requiere un esfuerzo considerable para convertir datos sin procesar en conocimiento. Por ejemplo, imagine que tiene una huerta en la que quiere plantar lechugas. Quiere optimizarla para poder cultivar la máxima cantidad de lechugas en el menor período de tiempo. Puede recopilar una gran cantidad de datos que influirán en el modo en que configura el entorno más eficaz para cultivar lechugas.

Como factores, puede considerar la exposición a la luz solar, la temperatura, la humedad del suelo y el aire, el tipo de lechuga y el origen de las semillas, la exposición al aire fresco, el tamaño del macetero y la calidad y la cantidad de tierra. La lista puede ser incluso más extensa, ya que podría haber factores que afecten al crecimiento y que ni siquiera conozca, como el nivel de ruido o el tipo de ruido junto a la huerta.

### Ciclo de vida de ciencia de datos

Si entiende el [ciclo de vida de la ciencia de datos](https://learn.microsoft.com/es-es/azure/architecture/data-science-process/lifecycle), podrá orientar mejor sus esfuerzos cuando cree nuevos conocimientos a partir de orígenes de datos.

![data-science-lifecycle](https://learn.microsoft.com/es-es/training/modules/introduction-rocket-launch-nasa/media/data-science-lifecycle.png)

Estos son los cuatro pasos del ciclo de vida de la ciencia de datos:

1. Definición de un objetivo empresarial mediante la experiencia en la materia
2. Recopilación, limpieza y manipulación de los datos
3. Elección de un algoritmo de aprendizaje automático y, después, entrenamiento y prueba del modelo
4. Implementación del modelo para usarlo con otras aplicaciones

<hr/>

## Establecimiento de un objetivo y obtención de experiencia

El **paso 1 del ciclo de vida de la ciencia de datos** consiste en definir un objetivo empresarial mediante la experiencia en la materia. Es fundamental que tenga un objetivo claro cuando empiece el análisis de los datos. En el ejemplo de la huerta, el objetivo consiste en generar el máximo de lechugas posible. Pero las condiciones de cultivo no son absolutas. Si la producción de línea base es 1 kilo de lechugas en 14 días, necesita una manera de acelerar la producción.

Ha oído que hablar con las plantas puede hacer que crezcan más rápido. Aunque parece *posible* que el sonido influya en el crecimiento de las plantas, la *probabilidad* de que el sonido afecte al resultado podría ser demasiado pequeña para que merezca la pena tenerla en cuenta. Aun así, decide experimentar. Descubre que, si reproduce música clásica a 50 decibelios durante 30 minutos cada 3 horas, obtiene el mismo kilo de lechugas en 13,5 días. En última instancia, incluir una programación musical en la huerta es una solución compleja que probablemente no merece la pena para reducir el tiempo de cultivo en medio día.

En el ejemplo de la huerta se demuestra que es importante tener acceso a expertos en la materia (SME) que conozcan los factores que afectan al tema que le ocupa. De este modo, no modificará variables que no aportarán cambios importantes.

Si aplica esta idea al lanzamiento de cohetes y los patrones meteorológicos, se dará cuenta de que el objetivo se divide en dos partes:

* Aumentar la probabilidad de que el día elegido para el lanzamiento tenga las condiciones meteorológicas adecuadas
* Saber qué condiciones deberían detener un lanzamiento

La experiencia necesaria para alcanzar estos objetivos reside en las contribuciones de meteorólogos, físicos, biólogos, ingenieros aeroespaciales y muchos más. Los expertos en la materia ayudan a definir los factores que pueden afectar a un lanzamiento y, por tanto, requieren especial atención, puesto que minimizan el número de variables que se deben analizar.

Por ejemplo, es posible que los expertos en la materia determinen que la cantidad de luz solar directa sobre la plataforma de lanzamiento no suponga una gran diferencia en el éxito del lanzamiento, pero que el porcentaje de humedad del aire sí. También es posible que sepan que hay intervalos de datos importantes que se pueden ignorar. Por ejemplo, si la temperatura del lugar de lanzamiento es inferior a -1,1 grados centígrados, los demás factores no se tienen en cuenta. No hay factores de mitigación cuando hace demasiado frío para lanzar un cohete de forma segura.

<hr/>

## Recopilación, limpieza y manipulación de los datos

El **paso 2 del ciclo de vida de la ciencia de datos** consiste en recopilar, limpiar y manipular los datos. Una vez que haya definido con claridad lo que quiere saber, puede evaluar los datos que tiene y los que es posible que tenga que recopilar. A partir de ahí, puede preparar los datos para que admitan la detección que le interesa.

### Obtención de datos

Con las restricciones, los ámbitos y la priorización de los datos que proporcionan los expertos en la materia, puede empezar a recopilar datos útiles. Este paso plantea sus propios desafíos. Si volvemos al ejemplo de la huerta, podría cultivar 10 lechugas bajo condiciones ligeramente diferentes y, después, determinar qué condiciones producen los mejores resultados.

En el caso del lanzamiento de un cohete, no es tan fácil hacer experimentos de comparación. Puede ejecutar simulaciones, pero se basan en el análisis de datos, no en un proceso literal de ensayo y error en condiciones exactas. No resulta ético ni económico realizar un lanzamiento de prueba bajo cada circunstancia única a fin de poder determinar con certeza las circunstancias más seguras. Además, muchas condiciones, como el tiempo, no se pueden controlar. (Aún así, es cierto que algunos de los datos que se usan en una simulación proceden de lanzamientos de cohetes fallidos que se intentaron en circunstancias negativas. De lo contrario, ¿cómo conocería las limitaciones de determinadas condiciones?) También puede usar otra información para determinar las restricciones, como la información recopilada de los aviones o cálculos físicos básicos o matemáticos.

### Limpieza y manipulación de los datos

A primera vista, que un paso del aprendizaje automático sea la *manipulación* de los datos le puede resultar sospechoso. En este caso, no significa que los datos se modifiquen para obtener el resultado deseado, sino que debe prestar atención para asegurarse de que los datos son la representación más precisa de la verdad.

Por ejemplo, con la huerta de lechugas, podría realizar un estudio centrado en la humedad del suelo. Recopila lecturas de humedad cada hora para poder determinar cómo afecta al crecimiento. Un día a las 14:55, el sensor de humedad deja de funcionar. Se da cuenta de que se ha estropeado y lo arregla antes de la lectura programada para las 16:00, pero pierde los datos que se habrían recopilado en la lectura de las 15:00. Es razonable manipular los datos y reemplazar el valor que falta por una media de las lecturas tomadas a las 14:00 y las 16:00. En cambio, si no descubre hasta el día siguiente que el sensor se ha estropeado, es posible que tenga más sentido *limpiar* los datos y quitar completamente del análisis las lecturas de ese día, para que los datos incompletos no lleven a un resultado inexacto.

Se necesita una gran cantidad de datos para predecir las condiciones ideales para el lanzamiento de un cohete. Es probable que la NASA tenga acceso a datos mejores de los que están disponibles públicamente. La NASA tiene acceso a conocimientos de expertos en la materia que analizan de cerca los matices de los lanzamientos de cohetes y las condiciones meteorológicas. También tiene acceso a todos los experimentos y análisis anteriores.

Por el contrario, en el modelo de aprendizaje automático que entrenará en el módulo siguiente de esta ruta de aprendizaje, se basará principalmente en datos meteorológicos accesibles, como la temperatura, las precipitaciones y la nubosidad. Se centrará en días pasados en los que se llevaron a cabo lanzamientos. El resultado realista es que este proyecto será menos preciso que las predicciones de la NASA. Como solo tiene ejemplos de lanzamientos correctos, el modelo de aprendizaje automático que entrena se sesgará hacia condiciones favorables.

<hr/>

## Selección de un algoritmo para entrenar y probar el modelo

El **paso 3 del ciclo de vida de la ciencia de datos** consiste en elegir un algoritmo de aprendizaje automático y, luego, entrenar y probar el modelo. En este punto del ciclo de vida de la ciencia de datos, tiene los datos que mejor representan la verdad sobre lo que investiga. Por tanto, es el momento de modelar el aprendizaje automático para empezar a descubrir conocimientos.

El *modelado* es el proceso de elegir qué características de datos es más probable que indiquen un conocimiento fiable. Estas características de datos pueden variar. Por ejemplo, podrían ser las columnas de una tabla, información secundaria como la diferencia entre dos columnas, o bien algo más sutil como el color de una imagen.

### Modelado

Para el huerto de lechugas, es probable que algunos aspectos del entorno sean más importantes que otros. Por ejemplo, la humedad del suelo es más importante que el nivel de ruido. Pero en el caso de otras características, puede ser difícil evaluar si una característica tiene una correlación más estrecha que otra con el resultado deseado. Por ejemplo, ¿la humedad del suelo es un mejor indicador del crecimiento en el tiempo que la temperatura? La *ingeniería de características* es una técnica que usa el modelo de aprendizaje automático para ayudarle a entender qué características se correlacionan más estrechamente con el resultado.

En el caso del lanzamiento de un cohete, no tiene acceso a algunos datos posiblemente muy correlacionados, como la forma, el tamaño y la clasificación de las nubes previstas en una fecha específica dentro de tres años. Pero tendrá tres fragmentos de datos principales que probablemente estén muy correlacionados: la temperatura, las precipitaciones y la humedad. En esta ruta de aprendizaje, el objetivo es usar datos de lanzamientos anteriores, datos meteorológicos anteriores y datos meteorológicos previstos para predecir si es probable que un lanzamiento se realice correctamente.

### Hoja de referencia rápida de algoritmos de aprendizaje automático

Un recurso práctico para determinar qué tipo de algoritmo de aprendizaje automático será útil para un análisis es la [hoja de referencia rápida de algoritmos de aprendizaje automático](https://learn.microsoft.com/es-es/azure/machine-learning/algorithm-cheat-sheet).

![algorithm-cheat-sheet](https://learn.microsoft.com/es-es/training/modules/introduction-rocket-launch-nasa/media/algorithm-cheat-sheet.png)

### Elección del algoritmo de aprendizaje automático correcto

Una vez más, la pregunta central es ¿*Permitirán las condiciones meteorológicas de un día concreto realizar con éxito el lanzamiento de un cohete?*.

La pregunta se responde con un *sí* o un *no*. Por lo tanto, es un problema en el que podría resultar útil un *algoritmo de clasificación de dos clases*. Si examina esa categoría en la hoja de referencia rápida de algoritmos, verá que puede elegir entre muchos. En este caso, un clasificador de árbol de decisión funcionaría bien. Este tipo de algoritmo toma observaciones sobre un evento, como las condiciones meteorológicas de un día concreto, y extrae conclusiones sobre el valor de destino. Su resultado es *sí* o *no* a la pregunta planteada.

### Entrenamiento y prueba de modelos de Machine Learning

Después de elegir el algoritmo de aprendizaje automático que usará, tiene que proporcionarle datos basados en la verdad. Cuando escriba datos complejos, querrá que el modelo genere la opción correcta. En este paso, se usa un conjunto de datos existente para entrenar el modelo.

En la unidad siguiente, veremos un ejemplo de identificación de frutos del bosque para describir cómo se puede entrenar a personas para aprender información nueva. Los modelos de aprendizaje automático son similares al experimento de identificación de frutos del bosque. Para entrenar el modelo, debe proporcionarle una entrada y una salida. Pero no le proporciona todos los datos porque, de lo contrario, el modelo se sobreajusta. En este caso, solo sabría identificar un subconjunto de datos posibles. No sería capaz de generalizar a nuevos elementos que son similares, pero diferentes. Por ese motivo, debe guardar algunos datos para probar el modelo. Para ello, solo debe proporcionarle los datos de entrada. Los datos de salida reales se usan para "evaluar" o "puntuar" el modelo.

Afortunadamente, los algoritmos de aprendizaje automático que necesita ya están escritos. También están disponibles las herramientas necesarias para dividir los datos, entrenar el modelo y probarlo. Puede acceder a estas herramientas y usarlas como un servicio, por lo que no necesita instalarlas en el equipo.

<hr/>

## Implementación del modelo de aprendizaje automático

El **paso 4 del ciclo de vida de la ciencia de datos** consiste en implementar el modelo de aprendizaje automático.

Por último, cuando el modelo esté entrenado y probado, podrá implementarlo no solo para usarlo en sus propios proyectos de ciencia de datos, sino para compartirlo con otros usuarios.

Cuando implemente el modelo, está disponible para su uso con otro software. Por ejemplo, puede crear un sitio web en el que un usuario cargue una imagen de un fruto del bosque y el sitio identifique de qué tipo se trata. Las posibilidades son enormes.

<hr/>

## Procedimiento de aprendizaje de las personas y los modelos de aprendizaje automático

Un modelo de aprendizaje automático se entrena de forma similar a como se entrenan las personas. Pero ¿cómo aprende un ser humano?

Imagine que pasea por una zona en la que hay cinco tipos diferentes de frutos del bosque que nunca había visto. Le piden que recoja 100 ejemplares aleatorios, incluido un fruto del bosque de cada una de las cinco especies nuevas. Le indican el nombre de los cinco tipos diferentes: frambuesa, arándano azul, mora, fresa y arándano rojo. Los otros 95 ejemplares que ha recogido pertenecen a uno de estos tipos.

Como puede asignar un nombre a los cinco tipos de frutos del bosque diferentes, está convencido de que podrá identificar los tipos de los 95 frutos del bosque restantes que ha recogido aleatoriamente. Es posible que algunas moras no estén maduras, por lo que serán más pequeñas y parecerán frambuesas, y que algunos arándanos azules no estén tan maduros y parezcan arándanos rojos. Aun así, cree que podría distinguir a qué tipo pertenece cada ejemplar y que sería capaz de clasificar los 100 frutos del bosque por tipo.

Después, le piden que recoja solo frambuesas en un campo nuevo adyacente. No duda del aspecto que tienen las frambuesas:

![raspberry](https://learn.microsoft.com/es-es/training/modules/introduction-rocket-launch-nasa/media/raspberry.png)

Completa esta tarea y consigue 10 frambuesas sin problema.

En resumen, los primeros 100 frutos del bosque se encontraban en su conjunto de datos inicial. Le proporcionaron una entrada (los 100 frutos del bosque) y una salida (los tipos de frutos del bosque incluidos) y le *entrenaron* para que pudiera identificar los ejemplares que recogió.

Después, le sometieron a una prueba. En un campo nuevo con frutos del bosque, dada cualquier entrada, debía identificar el tipo de fruto del bosque y seleccionar solo una salida: las frambuesas. Mientras caminaba entre las plantas, examinaba otros frutos del bosque (la entrada). Puso a prueba su modelo mental de frutos del bosque y solo eligió frambuesas. Llegado a este punto, está convencido de que su modelo mental de los frutos del bosque tiene una precisión del 100%.

Pero en ese momento, detecta un fruto del bosque que tiene un aspecto similar a una frambuesa, pero ligeramente diferente. Lo que no sabía es que, en realidad, había seis tipos de frutos del bosque en el nuevo campo. Encuentra más frambuesas, pero también recoge algunos de los otros frutos del bosque, pensando que podrían ser frambuesas, aunque tienen un aspecto algo diferente:

![thimbleberry](https://learn.microsoft.com/es-es/training/modules/introduction-rocket-launch-nasa/media/thimbleberry.png)

Tanto el nuevo ejemplar como la frambuesa son diferentes a los otros cuatro tipos de frutos del bosque, pero tienen un aspecto similar entre sí. Pero *no* son el mismo tipo de baya. El nuevo tipo de fruto del bosque que ha recogido es una *frambuesa salvaje*.

En este caso, el conjunto de datos no es lo suficientemente amplio. Sería poco preciso situar una frambuesa salvaje con las frambuesas simplemente porque cree que debería encajar en uno de los tipos de frutos del bosque y no sabe que hay más de cinco tipos diferentes. Cree que identifica los frutos del bosque con más precisión de la que demuestra en la realidad porque no sabe todo lo que necesita saber para ser preciso.

La identificación de los frutos del bosque podría parecer trivial, pero sus implicaciones se aplican a las soluciones de aprendizaje automático. Cuando estos tipos de soluciones afectan a la vida de las personas, como en el lanzamiento de un cohete, es necesario evitar estos tipos de errores en el análisis de datos.

<hr/>

## La ética en la ciencia de datos y el aprendizaje automático

Es importante comprender el papel que desempeña la ética en cada parte del ciclo de vida de la ciencia de los datos. Debe considerar la ética de las decisiones en cada paso. Comienza con una pregunta central y progresa a través de la disponibilidad del modelo.

En el ejemplo de los frutos del bosque, ha descubierto que en los conjuntos de datos de entrenamiento y prueba faltaba un fragmento de datos importante. No sabía nada de las frambuesas salvajes ni que había seis tipos de frutos del bosque, en lugar de solo cinco. Aunque la identificación de frutos del bosque puede parecer trivial, el fenómeno representa un problema mucho mayor. Además de la seguridad del lanzamiento de un cohete, la ausencia de estos datos puede *sesgar* los resultados e incluso poner vidas en riesgo. Por ejemplo, ¿sabía que hombres y mujeres presentan síntomas de ataques cardíacos totalmente distintos? En los estudios sanitarios recientes, se omitieron grandes poblaciones de personas de la recopilación de datos inicial, lo que afectó a los modelos de los síntomas de infarto que se usaban en la atención sanitaria.

### La ética y la seguridad del lanzamiento de cohetes

Los conocimientos y la experiencia de los colaboradores y los científicos de la NASA ayudan a garantizar la máxima probabilidad de que el lanzamiento de un cohete sea seguro y exitoso. Es posible que no tenga acceso a los mismos recursos, pero puede intentar ser lo más ético posible con los datos limitados disponibles.

En los módulos restantes de esta ruta de aprendizaje, verá cómo pueden ayudarle los datos meteorológicos disponibles públicamente a entender qué día es el adecuado para un lanzamiento correcto. El conjunto de datos con el que trabajará contiene información sobre 64 lanzamientos de cohetes con y sin tripulación. Con estos datos, puede consultar el tiempo durante esos 64 días para intentar obtener una descripción exacta de cuáles serían las condiciones meteorológicas adecuadas para garantizar el éxito del lanzamiento.

El conjunto de datos que usará solo contiene un lanzamiento incorrecto que se canceló debido al tiempo. Piense en el ejemplo de las frambuesas salvajes. Si no tiene una representación completa de los datos, no sabrá cuándo buscar nuevas categorías. En el ejemplo de los frutos del bosque, no sabía que había seis tipos diferentes y no identificó las frambuesas salvajes. En los datos de la NASA, faltan fechas de lanzamientos cancelados.

Los problemas de ciencia de datos requieren rigor e iteraciones. Con cada nuevo nivel de conocimiento que se obtiene de los datos, se descubre cuáles podrían faltar, qué preguntas nuevas formular y cómo se podrían priorizar los datos para entender con más precisión el mundo.

Los análisis que solo tienen en cuenta un ejemplo de factores negativos no aportan el tipo de datos que la NASA usaría cuando hay vidas en riesgo. Se necesitarán más datos y experiencia en la materia antes de que se pueda usar para cualquier tipo de toma de decisiones real. El conjunto de información con el que trabajará en los siguientes módulos de la ruta de aprendizaje proporciona una introducción al tipo de análisis que podría usarse como punto de partida.