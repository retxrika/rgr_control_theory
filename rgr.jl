using Pkg
# Для создания и обучения нейронных сетей.
#Pkg.add(PackageSpec(name="Flux", version="0.14.25"))
# Для моделирования и решения дифференциальных уравнений
#Pkg.add(PackageSpec(name="DifferentialEquations", version="7.15.0"))
# Для автоматического дифференцирования в задачах, связанных с моделированием.
#Pkg.add(PackageSpec(name="SciMLSensitivity", version="7.72.0"))
# Для методов оптимизации
#Pkg.add(PackageSpec(name="Optimisers", version="0.3.4"))
# Для сохранения и загрузки модели
#Pkg.add("BSON")
# Для вывода графиков
#Pkg.add("Plots")
#Pkg.status()
using Optimisers
using SciMLSensitivity
using DifferentialEquations
using Flux
using Random
using BSON
using Plots

# Параметры маятника.
m = Float32(0.2)   # Масса шарика (кг).
M = Float32(0.5)   # Масса тележки (кг).
L = Float32(0.3)   # Длина маятника (м).
g = Float32(9.81)  # Ускорение свободного падения (м/с^2).

# Критерий качества
function quality(teta_arr, p_arr)
    return sum(teta_arr .^ 2 + p_arr .^ 2) * 0.02
end

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

    return [sol[1, end], sol[2, end], sol[3, end], sol[4, end]], p[1]
end

function show_graphs(
    u0,
    show_x=true, 
    show_theta=true,
    show_crit=true
)
    # Генерация массива для оси X
    times = 0.0:0.02:100.0
    states_Tan_Yam7_theta = []
    states_Tan_Yam7_x = []
    u0_last = u0
    flag = false
    for i in 1:5001
        last_u4 = u0[4]
        if show_theta
            push!(states_Tan_Yam7_theta, u0[1])
        end
        if show_x        
            push!(states_Tan_Yam7_x, u0[3])
        end

        if !flag
            u0, p = test_model(Nothing, u0, TanYam7(), false, false)
        end

        if u0[1] > pi/2
            flag = true
            u0[1] = pi/2
            u0[4] = last_u4
        end
        if u0[1] < -pi/2
            flag = true
            u0[1] = -pi/2
            u0[4] = last_u4
        end
        
        if flag
            u0[3] += u0[4] * 0.02
        end
    end

    u0 = u0_last
    states_Vern7_theta = []
    states_Vern7_x = []
    flag = false

    for i in 1:5001
        last_u4 = u0[4]
        if show_theta
            push!(states_Vern7_theta, u0[1])
        end
        if show_x
            push!(states_Vern7_x, u0[3])
        end

        if !flag
            u0, p = test_model(Nothing, u0, Vern7(), false, false)
        end

        if u0[1] > pi/2
            flag = true
            u0[1] = pi/2
            u0[4] = last_u4
        end
        if u0[1] < -pi/2
            flag = true
            u0[1] = -pi/2
            u0[4] = last_u4
        end

        if flag
            u0[3] += u0[4] * 0.02
        end
    end

    # Построение графиков theta.
    if show_theta
        plot(times, states_Tan_Yam7_theta, label="Алгоритм TanYam7", xlabel="t", ylabel="Состояние theta", title="Графики зависимости состояний theta от времени")
        plot!(times, states_Vern7_theta, label="Алгоритм Vern7", color=:red)
        # Сохранение графиков.
        savefig("graphs_theta.png")
    end
    
    # Построение графиков для x.
    if show_x
        plot(times, states_Tan_Yam7_x, label="Алгоритм TanYam7", xlabel="t", ylabel="Состояние x", title="Графики зависимости состояний x от времени")
        plot!(times, states_Vern7_x, label="Алгоритм Vern7", color=:red)
        # Сохранение графиков.
        savefig("graphs_x.png")
    end

    # Критерий качества.
    if show_crit
        println("Критерий качества для Tan_Yam7: " * string(sum(states_Tan_Yam7_theta .^ 2) * 0.02))
        println("Критерий качества для Vern7: " * string(sum(states_Vern7_theta .^ 2) * 0.02))
    end
end

#Загружаем модель
model_path = "model.bson"
BSON.@load model_path model
println("Модель загружена из $model_path")

#Задаем начальное состояние
u0 = Float32[pi/6, 0, 0, 0]

# графики до изменения
show_graphs(u0)

u = u0
states_Tan_Yam7 = []
θ_Tan_Yam7 = []
X_Tan_Yam7 = []
P_Tan_Yam7 = []

# начальное значение
p = Float32(0)
num_iter = 5000
t_arr = collect(range(0, step=0.02, length=num_iter))

# Стабилизируем
count = 0
for i in 1:num_iter
    global u
    global count
    global p
    push!(states_Tan_Yam7, u)
    push!(θ_Tan_Yam7, u[1])
    push!(X_Tan_Yam7, u[2])
    push!(P_Tan_Yam7, p)

    # Применяем модель
    use = true

    # Если угол около нуля можно не применять модель
    if (-0.025 < u[1] < 0.025) &&  (-0.05 < u[2] < 0.05)
        use = false
        count += 1
    end

    u, p = test_model(model, u, TanYam7(), use, false)

    if !(-pi/2 < u[1]  < pi/2)
        println("θ = ", u[1], " p = ", p, " Сломалось на итерации ", i)
        break
    end
end

#Задаем начальное состояние
u0 = Float32[pi/6, 0, 0, 0]
u = u0
θ_Vern7 = []
X_Vern7 = []
P_Vern7 = []

# начальное значение
p = Float32(0)
num_iter = 5000

# график стабилизации
count = 0
for i in 1:num_iter
    global u
    global count
    global p
    push!(θ_Vern7, u[1])
    push!(X_Vern7, u[2])
    push!(P_Vern7, p)

    # Применяем модель
    use = true

    # Если угол около нуля можно не применять модель
    if (-0.025 < u[1] < 0.025) &&  (-0.05 < u[2] < 0.05)
        use = false
        count += 1
    end

    u, p = test_model(model, u, Vern7(), use, false)


    if !(-pi/2 < u[1]  < pi/2)
        println("θ = ", u[1], " p = ", p, " Сломалось на итерации ", i)
        break
    end
end

println("Критерий качества для Tan_Yam7: ", quality(θ_Tan_Yam7, P_Tan_Yam7))
println("Критерий качества для Vern7: ", quality(θ_Vern7, P_Vern7))

plot(t_arr, θ_Tan_Yam7, label="Изменение угла θ для Tan_Yam7", title="Стабилизация", xlabel="t", ylabel="θ", color=:red)
plot!(t_arr, θ_Vern7, label="Изменение угла θ для Vern7", color=:blue)

# сохранение графика
savefig("plot.png")

plot(t_arr, X_Tan_Yam7, label="Изменение траектории для Tan_Yam7", title="Траектория", xlabel="t", ylabel="X", color=:red)
plot!(t_arr, X_Vern7, label="Изменение траектории для Vern7", color=:blue)

savefig("traectory.png")

plot(t_arr, P_Tan_Yam7, label="Изменение силы для Tan_Yam7", title="Изменение силы", xlabel="t", ylabel="X", color=:red)
plot!(t_arr, P_Vern7, label="Изменение силы для Vern7", color=:blue)

savefig("p.png")
