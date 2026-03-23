using POMDPs
using MineralExploration
using POMCPOW
using Random
using GeoStats

# 启用 Plots 包，使用 Plotly 后端生成网页图
ENV["PLOTS_DEFAULT_BACKEND"] = "Plotly"
using Plots
plotly()

# ==========================================
# 0. 数据处理与模型初始化
# ==========================================
println("=== 1. 初始化模型与推断地质信念 ===")

real_easting  = [450120.0, 450800.0, 451500.0, 450300.0, 451800.0] 
real_northing = [3120500.0, 3121000.0, 3120100.0, 3121800.0, 3121500.0] 
my_quals = [0.45, 0.60, 0.55, 0.20, 0.15] 

min_E, max_E = 450000.0, 452000.0
min_N, max_N = 3120000.0, 3122000.0
span_E, span_N = max_E - min_E, max_N - min_N
grid_nx, grid_ny = 50, 50

my_coords = zeros(Int64, 2, length(real_easting))
for i in 1:length(real_easting)
    my_coords[1, i] = clamp(round(Int, (real_easting[i] - min_E) / span_E * (grid_nx - 1)) + 1, 1, grid_nx)
    my_coords[2, i] = clamp(round(Int, (real_northing[i] - min_N) / span_N * (grid_ny - 1)) + 1, 1, grid_ny)
end

my_data = RockObservations(ore_quals=my_quals, coordinates=my_coords)
m = MineralExplorationPOMDP(reservoir_dims=(span_E, span_N, 30.0), grid_dim=(grid_nx, grid_ny, 1), max_bores=10, delta=2, variogram=(0.005, 30.0, 0.0001), gp_mean=0.3, initial_data=my_data)

up = MEBeliefUpdater(m, 1000, 2.0)
MineralExploration.update!(up.geostats, my_data)
ds0 = POMDPs.initialstate(m)
MineralExploration.update!(ds0.gp_distribution, my_data)

println("正在进行粒子滤波推断...")
b0 = POMDPs.initialize_belief(up, ds0)
mean_ore, var_ore = MineralExploration.summarize(b0)

# ==========================================
# 1. 运行 AI 决策与批量推荐
# ==========================================
println("=== 2. 计算 AI 推荐坐标 ===")
next_action = NextActionSampler()
solver = POMCPOWSolver(tree_queries=1000, check_repeat_obs=true, check_repeat_act=true, next_action=next_action, k_action=2.0, alpha_action=0.25, k_observation=2.0, alpha_observation=0.1, criterion=POMCPOW.MaxUCB(100.0), final_criterion=POMCPOW.MaxQ(), estimate_value=0.0, rng=MersenneTwister(42))
planner = POMDPs.solve(solver, m)
best_action = POMDPs.action(planner, b0)

pomcp_rec_E = Float64[]
pomcp_rec_N = Float64[]
if best_action.type == :drill
    push!(pomcp_rec_E, min_E + (best_action.coords[1] - 1) * (span_E / (grid_nx - 1)))
    push!(pomcp_rec_N, min_N + (best_action.coords[2] - 1) * (span_N / (grid_ny - 1)))
end

candidates = []
for x in 1:grid_nx, y in 1:grid_ny
    if !any((my_coords[1, :] .== x) .& (my_coords[2, :] .== y))
        score = mean_ore[x, y, 1] + 2.0 * sqrt(var_ore[x, y, 1])
        push!(candidates, (x=x, y=y, score=score))
    end
end
sort!(candidates, by = cand -> cand.score, rev=true)

selected_points = []
for cand in candidates
    if !any(sqrt((cand.x - sp.x)^2 + (cand.y - sp.y)^2) < 5 for sp in selected_points)
        push!(selected_points, cand)
        length(selected_points) >= 5 && break
    end
end

batch_rec_E = [min_E + (p.x - 1) * (span_E / (grid_nx - 1)) for p in selected_points]
batch_rec_N = [min_N + (p.y - 1) * (span_N / (grid_ny - 1)) for p in selected_points]

# ==========================================
# 2. 全景画图模块：生成 4 张核心图表
# ==========================================
println("=== 3. 生成全景可视化图表 ===")
x_coords = range(min_E, max_E, length=grid_nx)
y_coords = range(min_N, max_N, length=grid_ny)

mean_matrix = mean_ore[:,:,1]' 
std_matrix = sqrt.(var_ore[:,:,1])'
# 计算 AI 打分矩阵 (品位 + 不确定性)
score_matrix = mean_matrix .+ 2.0 .* std_matrix

# 通用绘图参数
plot_args = (xlabel="Easting (m)", ylabel="Northing (m)", tickfont=font(7), titlefont=font(10, :bold), linewidth=0)

# 【图1】预测品位均值 (Predicted Mean)
p1 = contourf(x_coords, y_coords, mean_matrix; levels=20, color=:viridis, title="1. Predicted Grade (Where is the ore?)", plot_args...)
scatter!(p1, real_easting, real_northing, markershape=:star5, markersize=8, color=:red, label="Old Holes")

# 【图2】勘探不确定性 (Uncertainty/StdDev)
p2 = contourf(x_coords, y_coords, std_matrix; levels=20, color=:magma, title="2. Uncertainty (Where we know nothing)", plot_args...)
scatter!(p2, real_easting, real_northing, markershape=:star5, markersize=8, color=:red, label="Old Holes")

# 【图3】AI 综合打分热力图 (UCB Score) —— 解释 AI 为什么选这些点
p3 = contourf(x_coords, y_coords, score_matrix; levels=20, color=:turbo, title="3. AI Target Score (Mean + Uncertainty)", plot_args...)
scatter!(p3, real_easting, real_northing, markershape=:star5, markersize=6, color=:white, label="Old Holes")
if length(batch_rec_E) > 0
    scatter!(p3, batch_rec_E, batch_rec_N, markershape=:xcross, markersize=8, color=:black, markerstrokewidth=2, label="Top 5 Targets")
end
if length(pomcp_rec_E) > 0
    scatter!(p3, pomcp_rec_E, pomcp_rec_N, markershape=:star8, markersize=12, color=:lime, markerstrokecolor=:black, label="Best POMCP")
end

# 【图4】3D 矿脉地形图
p4 = surface(x_coords, y_coords, mean_matrix; color=:plasma, camera=(45, 45), title="4. 3D Grade Topography", zlabel="Grade", plot_args...)

# 拼合所有图表 (2行2列)
final_dashboard = plot(p1, p2, p3, p4, layout=(2, 2), size=(1200, 1000), margin=5Plots.mm)

savefig(final_dashboard, "All_Plots_Dashboard.html")
println("✅ 出图完成！请打开文件夹下的 【All_Plots_Dashboard.html】。")