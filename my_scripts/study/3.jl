using POMDPs
using MineralExploration
using POMCPOW
using Random
using GeoStats
using CSV           
using DataFrames    

# 启用 Plots 包，使用 Plotly 后端生成交互式网页图
ENV["PLOTS_DEFAULT_BACKEND"] = "Plotly"
using Plots
plotly()

# ==========================================
# 0. 读取数据与【随机隐藏功能】
# ==========================================
println("=== 0. 读取数据并进行随机盲测分割 ===")

df = CSV.read("C:\\Users\\Deep-Lei\\Desktop\\processed_drill_data.csv", DataFrame)
n_total = nrow(df)
println("总共加载了 $n_total 个真实钻孔数据。")

# 【核心新增】：随机隐藏数据逻辑
keep_ratio = 0.1  # 设定保留比例：0.5 表示保留 50% 给 AI 看，隐藏另外 50%
n_keep = round(Int, n_total * keep_ratio)

# 设定一个随机数种子保证每次隐藏的都是同一批孔（如果你想每次都随机，把这行注释掉）
Random.seed!(42) 
shuffled_indices = shuffle(1:n_total)

keep_indices = shuffled_indices[1:n_keep]
hidden_indices = shuffled_indices[n_keep+1:end]

# 提取给 AI 看的观测数据
obs_easting = df.X[keep_indices]
obs_northing = df.Y[keep_indices]
obs_quals = df.Average_Grade[keep_indices]

# 提取被隐藏起来的数据（仅用于最后画图验证）
hidden_easting = df.X[hidden_indices]
hidden_northing = df.Y[hidden_indices]
hidden_quals = df.Average_Grade[hidden_indices]

println("👉 盲测模式启动：提供给 AI $(length(obs_quals)) 个孔，偷偷隐藏了 $(length(hidden_quals)) 个孔作为验证集。")

# 计算全局边界框 (使用所有数据，确保地图大小不变)
padding = 50.0
min_E = minimum(df.X) - padding
max_E = maximum(df.X) + padding
min_N = minimum(df.Y) - padding
max_N = maximum(df.Y) + padding

span_E = max_E - min_E
span_N = max_N - min_N

cell_size = 20.0 
grid_nx = max(20, ceil(Int, span_E / cell_size)) 
grid_ny = max(20, ceil(Int, span_N / cell_size)) 

# 将观测数据映射到网格
obs_coords = zeros(Int64, 2, length(obs_easting))
for i in 1:length(obs_easting)
    norm_E = (obs_easting[i] - min_E) / span_E
    norm_N = (obs_northing[i] - min_N) / span_N
    obs_coords[1, i] = clamp(round(Int, norm_E * (grid_nx - 1)) + 1, 1, grid_nx)
    obs_coords[2, i] = clamp(round(Int, norm_N * (grid_ny - 1)) + 1, 1, grid_ny)
end

my_data = RockObservations(ore_quals=obs_quals, coordinates=obs_coords)

# ==========================================
# 1. 初始化 POMDP 环境与粒子滤波推断
# ==========================================
println("\n=== 1. AI 仅根据 $(keep_ratio*100)% 的已知数据推断地质信念 ===")

auto_gp_mean = sum(obs_quals) / length(obs_quals)

m = MineralExplorationPOMDP(
    reservoir_dims = (span_E, span_N, 30.0), 
    grid_dim = (grid_nx, grid_ny, 1),
    max_bores = 10,
    delta = 2,
    variogram = (0.005, 30.0, 0.0001), 
    gp_mean = auto_gp_mean,               
    initial_data = my_data       
)

up = MEBeliefUpdater(m, 1000, 2.0)
MineralExploration.update!(up.geostats, my_data)
ds0 = POMDPs.initialstate(m)
MineralExploration.update!(ds0.gp_distribution, my_data)

println("正在进行地质模拟...")
b0 = POMDPs.initialize_belief(up, ds0)
mean_ore, var_ore = MineralExploration.summarize(b0)

# ==========================================
# 2. 计算 AI 推荐坐标 (POMCP + UCB)
# ==========================================
println("\n=== 2. 强化学习计算 AI 推荐坐标 ===")
next_action = NextActionSampler()
solver = POMCPOWSolver(
    tree_queries = 1000, 
    check_repeat_obs = true, check_repeat_act = true,
    next_action = next_action, k_action = 2.0, alpha_action = 0.25,
    k_observation = 2.0, alpha_observation = 0.1,
    criterion = POMCPOW.MaxUCB(100.0), final_criterion = POMCPOW.MaxQ(),
    estimate_value = 0.0, rng = MersenneTwister(42)   
)

planner = POMDPs.solve(solver, m)
best_action = POMDPs.action(planner, b0)

pomcp_rec_E = Float64[]
pomcp_rec_N = Float64[]
if best_action.type == :drill
    rec_E = min_E + (best_action.coords[1] - 1) * (span_E / (grid_nx - 1))
    rec_N = min_N + (best_action.coords[2] - 1) * (span_N / (grid_ny - 1))
    push!(pomcp_rec_E, rec_E)
    push!(pomcp_rec_N, rec_N)
end

candidates = []
for x in 1:grid_nx, y in 1:grid_ny
    if !any((obs_coords[1, :] .== x) .& (obs_coords[2, :] .== y))
        score = mean_ore[x, y, 1] + 2.0 * sqrt(var_ore[x, y, 1])
        push!(candidates, (x=x, y=y, score=score))
    end
end
sort!(candidates, by = cand -> cand.score, rev=true)

min_grid_dist = 3.0 
selected_points = []
for cand in candidates
    if !any(sqrt((cand.x - sp.x)^2 + (cand.y - sp.y)^2) < min_grid_dist for sp in selected_points)
        push!(selected_points, cand)
        length(selected_points) >= 5 && break
    end
end

batch_rec_E = [min_E + (p.x - 1) * (span_E / (grid_nx - 1)) for p in selected_points]
batch_rec_N = [min_N + (p.y - 1) * (span_N / (grid_ny - 1)) for p in selected_points]

# ==========================================
# 3. 全景可视化看板 (对比隐藏的真实数据)
# ==========================================
println("\n=== 3. 生成盲测对比图表 ===")
x_coords = range(min_E, max_E, length=grid_nx)
y_coords = range(min_N, max_N, length=grid_ny)

mean_matrix = mean_ore[:,:,1]' 
std_matrix = sqrt.(var_ore[:,:,1])'
score_matrix = mean_matrix .+ 2.0 .* std_matrix

plot_args = (xlabel="Easting (m)", ylabel="Northing (m)", tickfont=font(7), titlefont=font(10, :bold), linewidth=0)

# 统一的图例标记函数
function add_markers!(p)
    # 给 AI 看见的红星
    scatter!(p, obs_easting, obs_northing, markershape=:star5, markersize=7, color=:red, markerstrokecolor=:white, label="Observed Holes")
    # 偷偷隐藏起来的白色圆圈
    scatter!(p, hidden_easting, hidden_northing, markershape=:circle, markersize=5, color=:white, markerstrokecolor=:black, markerstrokewidth=1, alpha=0.7, label="Hidden Truth")
end

p1 = contourf(x_coords, y_coords, mean_matrix; levels=20, color=:viridis, title="1. Predicted Grade (Mean)", plot_args...)
add_markers!(p1)

p2 = contourf(x_coords, y_coords, std_matrix; levels=20, color=:magma, title="2. Uncertainty (StdDev)", plot_args...)
add_markers!(p2)

p3 = contourf(x_coords, y_coords, score_matrix; levels=20, color=:turbo, title="3. AI Target Score & Blind Test", plot_args...)
add_markers!(p3)
if length(batch_rec_E) > 0
    scatter!(p3, batch_rec_E, batch_rec_N, markershape=:xcross, markersize=8, color=:black, markerstrokewidth=2, label="AI Top 5 Targets")
end
if length(pomcp_rec_E) > 0
    scatter!(p3, pomcp_rec_E, pomcp_rec_N, markershape=:star8, markersize=12, color=:lime, markerstrokecolor=:black, label="Best POMCP")
end

p4 = surface(x_coords, y_coords, mean_matrix; color=:plasma, camera=(45, 45), title="4. 3D Grade Topography", zlabel="Au Grade", plot_args...)

final_dashboard = plot(p1, p2, p3, p4, layout=(2, 2), size=(1200, 1000), margin=5Plots.mm)
savefig(final_dashboard, "All_Plots_Dashboard.html")
println("✅ 盲测结束！请打开【All_Plots_Dashboard.html】查收答卷。")