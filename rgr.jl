using Pkg
# Для создания и обучения нейронных сетей.
Pkg.add(PackageSpec(name="Flux", version="0.14.25"))
# Для моделирования и решения дифференциальных уравнений
Pkg.add(PackageSpec(name="DifferentialEquations", version="7.15.0"))
# Для автоматического дифференцирования в задачах, связанных с моделированием.
Pkg.add(PackageSpec(name="SciMLSensitivity", version="7.72.0"))
# Для методов оптимизации
Pkg.add(PackageSpec(name="Optimisers", version="0.3.4"))
Pkg.status()
using Optimisers
using SciMLSensitivity
using DifferentialEquations
using Flux
using Random

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
    #dω = (g * sin(θ) - cos(θ) * (force .+ m * L * ω^2 * sin(θ)) / (M .+ m)) / (L * (4/3 - m * cos(θ)^2 / (M .+ m)))
    
    # Линейная скорость тележки 𝑣 является производной положения 𝑥 (𝑑𝑥/𝑑𝑡=𝑣).
    dx = v
    #dv = (force .+ m * L * (ω^2 * sin(θ) - g * cos(θ)) / (M .+ m)) / (M .+ m)

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


# Задаем структуру нейронной сети
model = Chain(
    Dense(4, 32, relu),
    Dense(32, 64, relu),
    Dense(64, 64, relu),
    Dense(64, 32, relu),
    Dense(32, 1)  # Один выход - сила, применяемая к тележке
)

# Функция потерь
function loss(u0, target_angle)
    tspan = (Float32(0.0), Float32(0.02))
    loss = 0
    for u in u0
        p = model(u)
        prob = ODEProblem(cartpole!, u, tspan, p)
        sol = solve(prob)
        loss += abs(sol[1, end] - target_angle)/length(u0) 
    end   
    return loss
end

# Оптимизация
#opt_state = Flux.setup(Adam(), model)
#opt_state = Flux.Descent(learning_rate)
# Пример обучения
opt = Flux.Adam()

batch_size = 100
epochs = 20000

# Цель - вернуть в вертикальное положение
target_angle = 0.0

# Обучение
for epoch in 1:epochs
    global model
    global opt

    random_batch = generate_random_batch(batch_size, L)
    epoch_loss = 0
    
    gs = gradient(() -> loss(random_batch, target_angle), Flux.params(model))
    Flux.Optimise.update!(opt, Flux.params(model), gs)
    if epoch % 10000 == 0
        print("epoch_loss: ", loss(random_batch, target_angle))
        print("\n")
    end
end

u1 = Float32[pi/6, 0, 0, 0]
print("Начальный угол ", u1[1])
print("\n")
p = model(u1)
print("Предложенная сила воздействия F ", p[1])
print("\n")
tspan = (Float32(0.0), Float32(0.02))
prob = ODEProblem(cartpole!, u1, tspan, p)
sol = solve(prob)  # Решаем систему
print(sol[2, end], "\n")
print(sol[3, end], "\n")
print(sol[4, end], "\n")
print("угол после воздействия: ", sol[1, end][1])

u2 = Float32[sol[1, end], sol[2, end], sol[3, end], sol[4, end]]
p = model(u2)
print("Предложенная сила воздействия F ", p[1])
print("\n")
tspan = (Float32(0.0), Float32(0.02))
prob = ODEProblem(cartpole!, u2, tspan, p)
sol = solve(prob)  # Решаем систему
print(sol[2, end], "\n")
print(sol[3, end], "\n")
print(sol[4, end], "\n")
print("угол после воздействия: ", sol[1, end][1])