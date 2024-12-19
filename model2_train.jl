using Pkg
Pkg.add(PackageSpec(name="Flux", version="0.14.25"))
Pkg.add(PackageSpec(name="DifferentialEquations", version="7.15.0"))
Pkg.add(PackageSpec(name="SciMLSensitivity", version="7.72.0"))
Pkg.add(PackageSpec(name="Optimisers", version="0.3.4"))
Pkg.add("BSON")
Pkg.add("IterTools")
using Optimisers
using SciMLSensitivity
using DifferentialEquations
using Flux
using Random
using BSON
using IterTools
using Random

# генерация батча случайных элементов для обучения модели
function get_dataset(n, L)
    angle_values = Float32.(collect(range(-π/2, π/2, n)))
    ω_values = Float32.(collect(range(-100, 100, n)))
    x_values = Float32.(collect(range(-L, L, n)))
    v_values = Float32.(collect(range(-100, 100, n)))

    # Генерируем все комбинации
    combinations = collect(product(angle_values, ω_values, x_values, v_values))
    
    dataset = collect(shuffle(combinations))
    
    return dataset
end

# разделение датасета на обучающую, валидационную и тестовую выборки
function split_dataset(dataset, val_size, test_size)
    dataset_len = length(dataset)
    
    idx1 = round(Int, val_size * dataset_len)
    idx2 = round(Int, (val_size + test_size) * dataset_len)
    
    val_set = dataset[1:idx1]
    test_set = dataset[idx1+1:idx2]
    train_set = dataset[idx2+1:end]
    
    return train_set, val_set, test_set
end

# Уравнения движения
function cartpole!(du, u, p, t)
    θ, ω, x, v = u
    force = p[1]  # сила, которая будет применена к тележке

    # Уравнения движения
    dθ = ω
    #dω = (g * sin(θ) - cos(θ) * (force .+ m * L * ω^2 * sin(θ)) / (M .+ m)) / (L * (4/3 - m * cos(θ)^2 / (M .+ m)))
    
    dx = v
    #dv = (force .+ m * L * (ω^2 * sin(θ) - g * cos(θ)) / (M .+ m)) / (M .+ m)
    dv = (m * g * sin(θ) * cos(θ) - 7 / 3 * (force + m * L / 2 * ω^2 * sin(θ)))/(m * cos(θ)^2 - 7 /3 * M)
    dω = 3 / (7L / 2) * (g * sin(θ) - dv * cos(θ))

    du[1] = dθ
    du[2] = dω
    du[3] = dx
    du[4] = dv
end

# Функция потерь
function loss(set, target_angle)
    tspan = (Float32(0.0), Float32(0.02))
    loss = 0
    for u0 in set
        u0 = Float32[u0...]
        p = model(u0)
        prob = ODEProblem(cartpole!, u0, tspan, p)

        # решение дифф уравнения - получение конечного состояния тележки
        sol = solve(prob)

        # результирующий угол (минимизируется)
        theta1 = sol[1, end]
        loss += (theta1^Float32(2.0) + (p[1]^Float32(2.0))/200000) / length(set)
    end
    return loss
end

# Параметры маятника
m = Float32(0.2)   # масса шарика (кг)
M = Float32(0.5)   # масса тележки (кг)
L = Float32(0.3)   # длина маятника (м)
g = Float32(9.81)  # ускорение свободного падения (м/с^2)

# количество случайных значений каждого параметра для комбинаций
n = 20

# генерация датасета
dataset = get_dataset(n, L)
println("Размер датасета: ", length(dataset))

# доля валидационной выборки от всего датасета
val_size = 0.1

# доля тестовой выборки от всего датасета
test_size = 0.2

# разделение датасета на подвыборки
train_set, val_set, test_set = split_dataset(dataset, val_size, test_size)

println("Размер обучающей выборки: ", length(train_set))
println("Размер валидационной выборки: ", length(val_set))
println("Размер тестовой выборки: ", length(test_set))
print("\n")

# Задаем структуру нейронной сети
model = Chain(
    Dense(4, 64, leakyrelu),
    Dense(64, 128, leakyrelu),
    Dense(128, 64, leakyrelu),
    Dense(64, 32, leakyrelu),
    Dense(32, 1)
)

# Оптимизатор
opt = Flux.Adam()

# количество эпох
epochs = 50

# Цель - вернуть в вертикальное положение
target_angle = 0.0

# Размер батча
batch_size = 100

# Обучение
for epoch in 1:epochs
    train_loss = loss(train_set, target_angle)
    val_loss = loss(val_set, target_angle)
    println("Epoch $epoch, train loss: $train_loss, validation loss: $val_loss")

    # Перемешиваем данные для случайных мини-батчей
    indices = randperm(length(train_set))  # Индексы для перемешивания
    train_set_shuffled = train_set[indices]

    # Проходим по данным мини-батчами
    for i in 1:batch_size:length(train_set_shuffled)
        # Получаем текущий мини-батч
        batch = train_set_shuffled[i:min(i+batch_size-1, end)]

        # Вычисляем градиенты и обновляем веса
        gs = gradient(() -> loss(batch, target_angle), Flux.params(model))
        Flux.Optimise.update!(opt, Flux.params(model), gs)
    end
end

print("\n")
test_loss = loss(test_set, target_angle)
println("Test loss: $test_loss")
print("\n")

model_path = "model2.bson"

BSON.@save model_path model
println("Модель сохранена в $model_path")

BSON.@load model_path model
println("Модель загружена из $model_path")
print("\n")

println("Пример работы на случайных данных:")

u1 = Float32[rand(-π/2:π/2), rand(-100:100), rand(-L:L), rand(-100:100)]
println("Начальный угол: ", u1[1])

p = model(u1)
println("Предложенная сила воздействия F: ", p[1])

tspan = (Float32(0.0), Float32(0.02))
prob = ODEProblem(cartpole!, u1, tspan, p)
sol = solve(prob)

println(sol[2, end])
println(sol[3, end])
println(sol[4, end])
println("Угол после воздействия: ", sol[1, end][1])
