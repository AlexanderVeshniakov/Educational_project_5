### 1
- Функция compute_derivative на вход примет математическую функцию `y=f(x)` вида `x**2 - x + 21`  
- Вернёт производную: `2𝑥−1`

### 2
Функция compute_partial_derivative() на вход примет:
- математическую функцию `p = f(x, y)`зависящую от двух переменных вида `(x-y)**2`;
- переменную, по которой нужно посчитать частную производную.
Функция возвращает производную по выбранной переменной.

### 3
Предположим у нас есть:
- истинное значений `y_true`  
- предсказанное значение `y_pred`, описанное линейнным уравнением вида `y_pred = w*x + b`
1. Выведите формулу квадратного отклонения между истинным и предсказанным значениями. 
2. Вычислите частные производные полученной формулу квадратного отклонения по переменным `w` и `b`.

### 4
1. В этот раз тебе нужно написать класс `Gradient()`, который будет состоять из нескольких методов. Объект при 
инициализации получает два вектора: `X`, `Y`. Далее методы обращаются к ним уже внутри класса: 
     * `predict`: на вход она получает `w`, `b`, на выход она выдаёт прогнозные значения `Y_pred`;
     * `mse`: на вход она получает `Y_pred`, на выход выдает посчитанное MSE;
     * `update`: на вход она получает `w`, `b` и `a` (наш learning rate). Сделайте по умолчанию значение `a=0.0001`. А на \ 
выходe метод выдаёт новые значения `w`, `b`, которые обновились благодаря посчитанным градиентам.
2. Проверь работу методов, подав на вход класса два вектора `X`, `Y` и начальные параметры `w`, `b` в требуемые методы.

### 5
Модернизируем код  таким образом, чтобы появился метод `optimize`. На вход метод принимает `num_iterations`, 
`stopping_threshold=80`, `a=0.000001`. 
Метод должен итеративным образом пройти какое-то количество раз обновление значений `w`, `b`, 
придя к оптимальному значению. Критерии остановки:
  * если было превышено количество заранее заданных итераций `num_iterations`;
  * если новая итерация выдала разницу между текущей ошибкой и ошибкой прошлой итерации значение, меньшее, чем `stopping_threshold`;
  * метод должен вернуть финальные значения `w`, `b`, и `mse`.

_________________________________________________________________________________________________________________________________________________

### 1
- The input compute_derivative function will take the mathematical function 'y = f (x)' of the form 'x * * 2 - x + 21'
- Returns derivative: '2𝑥−1'

### 2
The compute_partial_derivative () function will accept:
- mathematical function 'p = f (x, y)' dependent on two variables of the form '(x-y) * * 2';
- is the variable by which you want to calculate the partial derivative.
The function returns a derivative of the selected variable.

### 3
Suppose we have:
- is the true value of'y _ true '
- is the predicted value 'y _ pred' described by a linine equation of the form 'y _ pred = w * x + b'
1. Derive the square deviation formula between the true and predicted values.
2. Calculate the partial derivatives of the resulting square deviation formula over the variables' w'and' b '.

### 4
1. This time you need to write the class 'Gradient ()', which will consist of several methods. Object at
initialization gets two vectors: 'X', 'Y'. Further, methods refer to them already inside the class:
* 'predict': at the input it receives 'w','b', at the output it produces predictive values ​of ' Y _ pred ';
* 'mse': it receives 'Y _ pred' at the input, gives the calculated MSE to the output;
* 'update': It receives 'w', 'b', and' a'as input (our learning rate). Make the default value'A = 0.0001 '. A on
the output of the method produces new values ​ ​ of 'w','b', which have been updated due to the calculated gradients.
2. Check the operation of the methods by supplying two vectors 'X', 'Y' and the initial parameters 'w', 'b' to the required methods to the class input.

### 5
We upgrade the code so that the'optimize' method appears. At the input, the method takes 'num_iterations',
`stopping_threshold=80`, `a=0.000001`.
The method must iteratively go through some number of times to update the values ​ ​ of 'w', 'b',
having reached the optimal value. Stopping criteria:
* if the number of predefined iterations 'num _ iterations' has been exceeded;
* if the new iteration returned a difference between the current error and the previous iteration error of less than 'stopping _ threshold';
* method must return the final values 'w','b', and 'mse'.