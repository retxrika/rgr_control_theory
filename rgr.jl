using Pkg
# Для создания и обучения нейронных сетей.
Pkg.add(PackageSpec(name="Flux", version="0.14.25"))
# Для моделирования и решения дифференциальных уравнений
Pkg.add(PackageSpec(name="DifferentialEquations", version="7.15.0"))
# Для автоматического дифференцирования в задачах, связанных с моделированием.
Pkg.add(PackageSpec(name="SciMLSensitivity", version="7.72.0"))
# Для методов оптимизации
Pkg.add(PackageSpec(name="Optimisers", version="0.3.4"))
Pkg.add("BSON")
Pkg.status()
using Optimisers
using SciMLSensitivity
using DifferentialEquations
using Flux
using Random
using BSON

# Параметры маятника.
m = Float32(0.2)   # Масса шарика (кг).
M = Float32(0.5)   # Масса тележки (кг).
L = Float32(0.3)   # Длина маятника (м).
g = Float32(9.81)  # Ускорение свободного падения (м/с^2).

#= 
Генерация батча (набора) случайных элементов для обучения модели.

Аргументы:
N — количество элементов (начальных состояний) в батче.
L — максимальная длина диапазона для положения тележки (в метрах).
=#
function generate_random_batch(N, L)
    # Сгенерированные начальные состояния.
    elements = []
    # Количество дискретных значений, которые используются для 
    # генерации возможных начальных состояний системы.
    # Определяет, насколько подробно делится диапазон каждого 
    # параметра (угол, угловая скорость, положение, линейная скорость) 
    # на отдельные точки.
    n_values = 20
    # Шаг изменения угла маятника в диапазоне [-π/2, π/2].
    angle_step = π / n_values
    # Шаг изменения угловой скорости в диапазоне [-100, 100].
    ω_step = 100 * 2 / n_values
    # Шаг изменения положения тележки в диапазоне [-L, L].
    x_step = L * 2 / n_values
    #v_step = 100*2/n_values

    # Список значений угла маятника от -π/2 до π/2 с шагом angle_step.
    angle_values = collect(-π/2 : angle_step : π/2)
    # Список значений угловой скорости от -100 до 100 с шагом ω_step.
    ω_values = collect(-100 : ω_step : 100)
    # Список значений положения тележки от -L до L с шагом x_step.
    x_values = collect(-L : x_step : L)

    # Цикл повторяется N раз (по количеству элементов в батче).
    for _ in 1:N
        # Генерация случайного положения тележки.

        # Выбираем случайное число из массива.
        rand_angle = rand(angle_values)
        rand_ω = rand(ω_values)
        rand_x = rand(x_values)
        #v[-100 100] 
        rand_v = rand(ω_values)
        
        # Вектор начального состояния u0 в формате Float32:
        # rand_angle: Угол маятника.
        # rand_ω: Угловая скорость маятника.
        # rand_x: Положение тележки.
        # rand_v: Линейная скорость тележки.
        u0 = Float32[rand_angle, rand_ω, rand_x, rand_v]
        # Вектор u0 добавляется в массив elements.
        push!(elements, u0)
    end

    # После выполнения всех итераций цикл возвращает массив elements, 
    # содержащий N случайных начальных состояний системы.
    return elements
end

#= 
Уравнения движения. Описывает динамическую модель системы "маятник на тележке" 
в виде уравнений движения.

Аргументы:
du: Вектор, куда записываются производные параметров (результат вычислений).
    Например, du = [dθ, dω, dx, dv], где:
    𝑑θ/𝑑𝑡 = ω,
    𝑑𝜔/𝑑𝑡 = угловое ускорение маятника,
    𝑑𝑥/𝑑𝑡 = 𝑣,
    𝑑𝑣/𝑑𝑡 = линейное ускорение тележки.
u:  Вектор текущего состояния системы, содержащий:
    θ — угол маятника относительно вертикали (в радианах).
    ω — угловую скорость маятника (рад/с).
    x — положение тележки (м).
    v — линейную скорость тележки (м/с). 
p:  Параметры внешнего воздействия. Здесь это вектор с единственным элементом:
    force (𝐹) — сила, приложенная к тележке.
t:  Время. В данном случае время явно не используется в уравнениях, но передается 
    для совместимости с численными методами решения ОДУ (обыкновенные дифференциальные 
    уравнения).
=#
function cartpole!(du, u, p, t)
    # Из вектора текущего состояния извлекаются значения угла, угловой скорости, положения и скорости.
    θ, ω, x, v = u
    force = p[1]  # сила, которая будет применена к тележке

    # Уравнения движения.
    # Угловая скорость маятника 𝜔 является производной угла 𝜃 (𝑑𝜃/𝑑𝑡=ω).
    dθ = ω
    
    # Линейная скорость тележки 𝑣 является производной положения 𝑥 (𝑑𝑥/𝑑𝑡=𝑣).
    dx = v

    # Линейное ускорение тележки (𝑑𝑣/𝑑𝑡).
    dv = (m * g * sin(θ) * cos(θ) - 7 / 3 * (force + m * L / 2 * ω^2 * sin(θ)))/(m * cos(θ)^2 - 7 /3 * M)
    #= 
    Угловое ускорение маятника (𝑑𝜔/𝑑𝑡), зависит от:
    силы тяжести (𝑔), 
    угла маятника (sin(𝜃)), 
    компоненты линейного ускорения тележки (𝑑𝑣), которая влияет на движение маятника через силу реакции.
    =#
    dω = 3 / (7L / 2) * (g * sin(θ) - dv * cos(θ))

    # Запись производных в du.
    du[1] = dθ
    du[2] = dω
    du[3] = dx
    du[4] = dv
end


function test_model(model, u0, alg, use_model, print_u=true)
    
    p = [0]
    # Получаем силу от модели.
    if use_model
        p = model(u0)
    end
    
    
    # Время воздействия на тележку.
    tspan = (Float32(0.0), Float32(0.02))
    
    # При помощи ODEProblem получаем задачу дифф. ур-ий.
    prob = ODEProblem(cartpole!, u0, tspan, p)
    
    # Решаем систему.
    sol = solve(prob, alg)
    if print_u
        println("Начальный угол ", u0[1])
        println("Предложенная сила воздействия F ", p[1])
        println("Угол θ после воздействия: ", sol[1, end][1])
        println("Скорость изменения угла ω после воздействия: ", sol[2, end])
        println("Положение тележки x после воздействия: ", sol[3, end])
        println("Скорость тележки v после воздействия: ", sol[4, end])
    end

    return [sol[1, end], sol[2, end], sol[3, end], sol[4, end]], p
end




#Загружаем модель
model_path = "model.bson"
BSON.@load model_path model
println("Модель загружена из $model_path")

#Задаем начальное состояние
u0 = Float32[pi/6, 0, 0, 0]
u = u0
states_Tan_Yam7 = []
# Стабилизируем
count = 0
for i in 1:5000
    global u
    global count
    push!(states_Tan_Yam7, u)

    # Применяем модель
    use = true

    # Если угол около нуля можно не применять модель
    if (-0.025 < u[1] < 0.025) &&  (-0.05 < u[2] < 0.05)
        use = false
        count += 1
    end

    u, p = test_model(model, u, TanYam7(), use)


    if !(-pi/2 < u[1]  < pi/2)
        println("θ = ", u[1], " p = ", p, " Сломалось на итерации ", i)
        break
    end
end
println(count)
#=
#Загружаем модель
model_path = "model.bson"
BSON.@load model_path model
println("Модель загружена из $model_path")

#Задаем начальное состояние
u0 = Float32[pi/6, 0, 0, 0]
u = u0
states_Vern7 = []
# Стабилизируем
for i in 1:5000
    global u
    push!(states_Vern7, u)
    u, p = test_model(model, u, Vern7(lazy=false))
    if !(-pi/2 < u[1]  < pi/2)
        println("θ = ", u[1], " p = ", p, " Сломалось на итерации ", i)
        break
    end
end
=#

#= TO DO:
1 Построить графики u[1], u[2], u[3], u[4] от времени временной интервал
0.00, 0.02, 0.04, ..... 100.00 (5000 итераций) для каждой модели (Должны быть 4 картинки по 2 графика для каждого алгоритма)

1.1 Обернуть в цикл/функцию всё что идет от Загружаем модель до TO DO, потому что дальше надо будет повторить

2 Построить такой же цикл на 5000 итераций, но чтобы модель не применялась
(в функции test_model use_model всегда = false) мб добавить флаг, чтобы не писать вторую функцию

3 Для цикла где модель не применяется построить θ от t и x от t
=#


