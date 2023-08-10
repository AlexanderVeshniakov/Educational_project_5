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

