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
# 0. 读取数据与自动计算动态网格
# ==========================================
println("=== 0. 读取真实钻孔数据并自动生成网格 ===")

# 读取 Python 处理好的 CSV 文件
df = CSV.read("C:\\Users\\Deep-Lei\\Desktop\\processed_drill_data.csv", DataFrame)

real_easting  = df.X
real_northing = df.Y
my_quals      = df.Average_Grade

println("成功加载 $(length(my_quals)) 个真实钻孔数据！")

# 1. 自动计算边界框 (加上 50 米缓冲，防止钻孔贴边)
padding = 50.0
min_E = minimum(real_easting) - padding
max_E = maximum(real_easting) + padding
min_N = minimum(real_northing) - padding
max_N = maximum(real_northing) + padding

span_E = max_E - min_E
span_N = max_N - min_N
println("自动侦测矿区跨度: 东西向 $(round(span_E, digits=2))m, 南北向 $(round(span_N, digits=2))m")

# 2. 【核心突破】：动态计算网格数量
cell_size = 20.0 # 设定网格分辨率：每个格子代表 20m x 20m (你可以根据算力修改，越小越精细但也越慢)
grid_nx = max(20, ceil(Int, span_E / cell_size)) # 算出 X 方向需要多少个格子
grid_ny = max(20, ceil(Int, span_N / cell_size)) # 算出 Y 方向需要多少个格子

println("按照 $(cell_size)m/格 的分辨率，AI 自动生成的计算网格为: $(grid_nx) × $(grid_ny)")

# 3. 映射到动态网格
my_coords = zeros(Int64, 2, length(real_easting))
for i in 1:length(real_easting)
    norm_E = (real_easting[i] - min_E) / span_E
    norm_N = (real_northing[i] - min_N) / span_N
    my_coords[1, i] = clamp(round(Int, norm_E * (grid_nx - 1)) + 1, 1, grid_nx)
    my_coords[2, i] = clamp(round(Int, norm_N * (grid_ny - 1)) + 1, 1, grid_ny)
end

my_data = RockObservations(ore_quals=my_quals, coordinates=my_coords)

# ==========================================
# 1. 初始化 POMDP 环境与粒子滤波推断
# ==========================================
println("\n=== 1. 初始化模型与推断地质信念 ===")

# 自动推断先验均值
auto_gp_mean = sum(my_quals) / length(my_quals)

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

println("正在进行地质模拟 (若网格大于 80x80，这步可能需要 1~2 分钟)...")
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
    println("⭐ [AI 单步绝对最优点]: (Easting: $(round(rec_E, digits=2)), Northing: $(round(rec_N, digits=2)))")
end

# UCB 批量备选
candidates = []
for x in 1:grid_nx, y in 1:grid_ny
    if !any((my_coords[1, :] .== x) .& (my_coords[2, :] .== y))
        score = mean_ore[x, y, 1] + 2.0 * sqrt(var_ore[x, y, 1])
        push!(candidates, (x=x, y=y, score=score))
    end
end
sort!(candidates, by = cand -> cand.score, rev=true)

# 控制备选点不要靠得太近 (间距必须大于3个网格)
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
# 3. 全景可视化看板 (Dashboard)
# ==========================================
println("\n=== 3. 生成高级全景可视化看板 ===")
x_coords = range(min_E, max_E, length=grid_nx)
y_coords = range(min_N, max_N, length=grid_ny)

mean_matrix = mean_ore[:,:,1]' 
std_matrix = sqrt.(var_ore[:,:,1])'
score_matrix = mean_matrix .+ 2.0 .* std_matrix

plot_args = (xlabel="Easting (m)", ylabel="Northing (m)", tickfont=font(7), titlefont=font(10, :bold), linewidth=0)

p1 = contourf(x_coords, y_coords, mean_matrix; levels=20, color=:viridis, title="1. Predicted Grade (Mean)", plot_args...)
scatter!(p1, real_easting, real_northing, markershape=:star5, markersize=6, color=:red, label="Old Holes")

p2 = contourf(x_coords, y_coords, std_matrix; levels=20, color=:magma, title="2. Uncertainty (StdDev)", plot_args...)
scatter!(p2, real_easting, real_northing, markershape=:star5, markersize=6, color=:red, label="Old Holes")

p3 = contourf(x_coords, y_coords, score_matrix; levels=20, color=:turbo, title="3. AI Target Score (UCB)", plot_args...)
scatter!(p3, real_easting, real_northing, markershape=:star5, markersize=4, color=:white, label="Old Holes")
if length(batch_rec_E) > 0
    scatter!(p3, batch_rec_E, batch_rec_N, markershape=:xcross, markersize=8, color=:black, markerstrokewidth=2, label="Top 5 Targets")
end
if length(pomcp_rec_E) > 0
    scatter!(p3, pomcp_rec_E, pomcp_rec_N, markershape=:star8, markersize=12, color=:lime, markerstrokecolor=:black, label="Best POMCP")
end

p4 = surface(x_coords, y_coords, mean_matrix; color=:plasma, camera=(45, 45), title="4. 3D Grade Topography", zlabel="Au Grade", plot_args...)

final_dashboard = plot(p1, p2, p3, p4, layout=(2, 2), size=(1200, 1000), margin=5Plots.mm)
savefig(final_dashboard, "All_Plots_Dashboard.html")
println("✅ 流程全部结束！你的动态网格版工程图表已生成：【All_Plots_Dashboard.html】。")