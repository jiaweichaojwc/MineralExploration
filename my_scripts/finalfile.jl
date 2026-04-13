#=
需要的输入是个文件夹，文件夹的里包含多个csv文件，每个csv文件包含数据如下格式

ID	X	Y	Average_Grade
钻井 1,B20,3100-3060M	509850.934	4675987.059	20
钻井 5,B20,3170-3120M	509863.036	4676189.237	20
钻井 8,B25,3360-3290M	509744.967	4676191.294	25
钻井 9,B18,3500-3440M	509802.622	4676291.345	18
钻井 10,B28,3580-3510M	509625.248	4676192.241	28
钻井 11,B15,3660-3600M	509680.428	4676292.286	15
钻井 12,B18,3620-3550M	509553.514	4676121.053	18
钻井 14,B15,3530-3490M	509715.993	4676246.793	15
钻井 15,B20,3690-3600M	509531.96	4676185.449	20
钻井 20,B15,3700-3670M	510814.974	4676740.471	15

其中X,Y是国家2000坐标系的坐标，Average_Grade是平均品位。每个csv文件的格式相同，但数据不同。需要对每个csv文件进行处理，输出一个txt文件和两张图（png和html）。txt文件包含推荐的打孔坐标，png和html图展示预测的品位分布、预测不确定性分布、AI推荐靶区分布等信息。所有结果保存在一个以csv文件名命名的独立文件夹中，避免覆盖。
=#

using POMDPs
using MineralExploration
using POMCPOW
using Random
using GeoStats
using CSV           
using DataFrames    
using Dates
using Plots

# ==========================================
# 🌟 全局函数定义（避免循环内重复定义警告）
# ==========================================
function add_markers!(p, obs_easting, obs_northing, hidden_easting, hidden_northing)
    scatter!(p, obs_easting, obs_northing, 
        markershape=:star5, markersize=3.5, color=:red, 
        markerstrokecolor=:black, markerstrokewidth=0.5, label="Observed")
    if length(hidden_easting) > 0
        scatter!(p, hidden_easting, hidden_northing, 
            markershape=:circle, markersize=2.5, color=:white, 
            markerstrokecolor=:black, markerstrokewidth=0.5, alpha=0.8, label="Hidden")
    end
end

function add_ai_targets!(p, batch_rec_E, batch_rec_N, pomcp_rec_E, pomcp_rec_N)
    if length(batch_rec_E) > 0
        scatter!(p, batch_rec_E, batch_rec_N, 
            markershape=:xcross, markersize=7, color=:black, 
            markerstrokewidth=2, label="Top Targets")
    end
    if length(pomcp_rec_E) > 0
        scatter!(p, pomcp_rec_E, pomcp_rec_N, 
            markershape=:star8, markersize=13, color=:lime, 
            markerstrokecolor=:black, markerstrokewidth=1, label="Best POMCP")
    end
end

# ==========================================
# 🌟 只需修改这里：文件夹路径 🌟
# ==========================================
input_folder = raw"C:\Users\Deep-Lei\Desktop\data\内蒙古锡林郭勒盟油气-3.023km2【油气】（费）（任务，下载）\智能布钻\转化后格式" # 存放CSV的文件夹
output_base_dir = raw"C:\\Users\\Deep-Lei\\Desktop"            # 结果输出根目录

# 2. 核心计算参数
cell_size = 10                # 网格分辨率 (米/格)
padding = 50.0                # 矿区边界外扩缓冲距离 (米)
min_grid_dist = 1           # 批量推荐打孔点之间的最小网格距离限制
max_target_num = 800          # 单文件最多输出的备选靶区数量

# 3. 盲测(交叉验证)参数
keep_ratio = 1                # 1.0 表示全量数据用于AI计算
random_seed = 42              # 随机种子，保证结果可复现

# ==========================================
# 🔧 批量读取逻辑 (只用标准库，无需安装新包)
# ==========================================
# 获取文件夹下所有文件
all_files = readdir(input_folder, join=true)
# 筛选出 .csv 结尾的文件
csv_files = filter(f -> endswith(lowercase(f), ".csv"), all_files)

if isempty(csv_files)
    error("❌ 文件夹里没有找到 CSV 文件！请检查路径：$input_folder")
end
println("=== 📂 找到 $(length(csv_files)) 个 CSV 文件，开始处理... ===")

# ==========================================
# 🔄 主循环：对每个 CSV 执行完整计算流程
# ==========================================
for csv_file_path in csv_files
    
    # --- 获取当前文件名 (用于生成独立文件夹，防止覆盖) ---
    file_name = splitext(basename(csv_file_path))[1]
    println("\n🚀 正在处理: $file_name ...")

    # --- 为当前文件创建独立的时间戳结果文件夹 ---
    timestamp = Dates.format(Dates.now(), "yyyymmdd_HHMMSS")
    folder_name = joinpath(output_base_dir, "Result_$(file_name)_$timestamp")
    mkpath(folder_name)
    println("=== 📁 结果保存至: $folder_name ===")

    # ==========================================
    # 1. 读取数据、盲测分割与动态网格
    # ==========================================
    println("=== 1. 读取数据并进行盲测分割 ===")
    df = CSV.read(csv_file_path, DataFrame)
    # 备份原始数据
    cp(csv_file_path, joinpath(folder_name, "backup_$(file_name).csv"), force=true)
    n_total = nrow(df)
    println("总共加载了 $n_total 个真实钻孔数据。")

    # 盲测随机隐藏逻辑
    n_keep = round(Int, n_total * keep_ratio)
    if random_seed != -1
        Random.seed!(random_seed)
    end
    shuffled_indices = shuffle(1:n_total)
    keep_indices = shuffled_indices[1:n_keep]
    hidden_indices = shuffled_indices[n_keep+1:end]

    obs_easting = df.X[keep_indices]
    obs_northing = df.Y[keep_indices]
    obs_quals = df.Average_Grade[keep_indices]
    hidden_easting = df.X[hidden_indices]
    hidden_northing = df.Y[hidden_indices]
    hidden_quals = df.Average_Grade[hidden_indices]

    println("👉 盲测模式: 提供给 AI $(length(obs_quals)) 个孔，隐藏了 $(length(hidden_quals)) 个孔作为验证。")

    # 自动计算全局边界框
    min_E = minimum(df.X) - padding
    max_E = maximum(df.X) + padding
    min_N = minimum(df.Y) - padding
    max_N = maximum(df.Y) + padding
    span_E = max_E - min_E
    span_N = max_N - min_N

    # 【防内存溢出机制】最大允许 60x60 网格
    max_grid_dim = 150
    current_cell_size = cell_size 
    if (span_E / current_cell_size) > max_grid_dim || (span_N / current_cell_size) > max_grid_dim
        safe_cell_size = max(span_E / max_grid_dim, span_N / max_grid_dim)
        println("⚠️ 警告: 矿区跨度过大，已启动防爆内存保护！")
        current_cell_size = safe_cell_size
        println("🛡️ 系统已自动将网格分辨率动态降级为: $(round(current_cell_size, digits=1)) 米/格。")
    end

    grid_nx = max(20, ceil(Int, span_E / current_cell_size)) 
    grid_ny = max(20, ceil(Int, span_N / current_cell_size)) 
    println("👉 最终使用的 AI 运算网格大小为: $(grid_nx) × $(grid_ny)")

    # 将观测数据映射到动态网格
    obs_coords = zeros(Int64, 2, length(obs_easting))
    for i in 1:length(obs_easting)
        norm_E = (obs_easting[i] - min_E) / span_E
        norm_N = (obs_northing[i] - min_N) / span_N
        obs_coords[1, i] = clamp(round(Int, norm_E * (grid_nx - 1)) + 1, 1, grid_nx)
        obs_coords[2, i] = clamp(round(Int, norm_N * (grid_ny - 1)) + 1, 1, grid_ny)
    end
    my_data = RockObservations(ore_quals=obs_quals, coordinates=obs_coords)

    # ==========================================
    # 2. 初始化模型与粒子滤波推断
    # ==========================================
    println("\n=== 2. AI 地质信念推断 ===")
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

    println("正在进行地质模拟 (可能需要几十秒)...")
    b0 = POMDPs.initialize_belief(up, ds0)
    mean_ore, var_ore = MineralExploration.summarize(b0)

    # ==========================================
    # 3. 计算 AI 推荐坐标 (POMCP + UCB)
    # ==========================================
    println("\n=== 3. 强化学习计算推荐坐标 ===")
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

    # POMCP 单步最优靶区
    pomcp_rec_E = Float64[]
    pomcp_rec_N = Float64[]
    pomcp_text = ""
    if best_action.type == :drill
        rec_E = min_E + (best_action.coords[1] - 1) * (span_E / (grid_nx - 1))
        rec_N = min_N + (best_action.coords[2] - 1) * (span_N / (grid_ny - 1))
        push!(pomcp_rec_E, rec_E)
        push!(pomcp_rec_N, rec_N)
        pomcp_text = "✅ [POMCP 单步最优靶区]: (Easting: $(round(rec_E, digits=2)), Northing: $(round(rec_N, digits=2)))"
        println("⭐ ", pomcp_text)
    end

    # UCB 批量备选靶区
    candidates = []
    for x in 1:grid_nx, y in 1:grid_ny
        if !any((obs_coords[1, :] .== x) .& (obs_coords[2, :] .== y))
            score = mean_ore[x, y, 1] + 2.0 * sqrt(var_ore[x, y, 1])
            push!(candidates, (x=x, y=y, score=score))
        end
    end
    sort!(candidates, by = cand -> cand.score, rev=true)

    # 筛选满足最小间距的靶区
    selected_points = []
    for cand in candidates
        if !any(sqrt((cand.x - sp.x)^2 + (cand.y - sp.y)^2) < min_grid_dist for sp in selected_points)
            push!(selected_points, cand)
            length(selected_points) >= max_target_num && break
        end 
    end

    batch_rec_E = [min_E + (p.x - 1) * (span_E / (grid_nx - 1)) for p in selected_points]
    batch_rec_N = [min_N + (p.y - 1) * (span_N / (grid_ny - 1)) for p in selected_points]
    println("✅ 共筛选出 $(length(batch_rec_E)) 个满足间距要求的备选靶区")

    # ==========================================
    # 4. 生成可视化图表
    # ==========================================
    x_coords = range(min_E, max_E, length=grid_nx)
    y_coords = range(min_N, max_N, length=grid_ny)
    mean_matrix = mean_ore[:,:,1]' 
    std_matrix = sqrt.(var_ore[:,:,1])'
    score_matrix = mean_matrix .+ 2.0 .* std_matrix

    # 4.1 高清静态PNG图
    println("\n=== 4.1 正在生成高清静态图 (PNG) ===")
    gr(dpi=300)
    png_args = (
        xlabel="Easting (m)", ylabel="Northing (m)", 
        tickfont=font(7, "Arial"), xrotation=45,                        
        titlefont=font(11, "Arial", :bold), guidefont=font(9, "Arial", :bold),
        linewidth=0, framestyle=:box,
        bottom_margin=12Plots.mm, left_margin=10Plots.mm, right_margin=22Plots.mm,
        colorbar_tickfont=font(7, "Arial")
    )

    p1_png = contourf(x_coords, y_coords, mean_matrix; levels=20, color=:terrain, title="(A) Predicted Au Grade (Mean)", png_args...)
    add_markers!(p1_png, obs_easting, obs_northing, hidden_easting, hidden_northing)

    p2_png = contourf(x_coords, y_coords, std_matrix; levels=20, color=:inferno, title="(B) Exploration Uncertainty", png_args...)
    add_markers!(p2_png, obs_easting, obs_northing, hidden_easting, hidden_northing)

    p3_png = contourf(x_coords, y_coords, score_matrix; levels=20, color=:jet, title="(C) AI Target Score", png_args...)
    add_markers!(p3_png, obs_easting, obs_northing, hidden_easting, hidden_northing)
    add_ai_targets!(p3_png, batch_rec_E, batch_rec_N, pomcp_rec_E, pomcp_rec_N)

    p4_png = surface(x_coords, y_coords, mean_matrix; color=:terrain, camera=(35, 45), title="(D) 3D Grade Topography", zlabel="Au Grade", png_args...)

    png_dashboard = plot(p1_png, p2_png, p3_png, p4_png, layout=(2, 2), size=(1920, 1200), margin=18Plots.mm, background_color=:white)
    savefig(png_dashboard, joinpath(folder_name, "HighRes_$(file_name).png"))

    # 4.2 交互HTML图
    println("=== 4.2 正在生成交互网页图 (HTML) ===")
    plotly()
    html_args = (
        xlabel="Easting (m)", ylabel="Northing (m)", 
        tickfont=font(9), titlefont=font(12, :bold), linewidth=0
    )

    p1_html = contourf(x_coords, y_coords, mean_matrix; levels=20, color=:terrain, title="(A) Predicted Au Grade (Mean)", html_args...)
    add_markers!(p1_html, obs_easting, obs_northing, hidden_easting, hidden_northing)

    p2_html = contourf(x_coords, y_coords, std_matrix; levels=20, color=:inferno, title="(B) Exploration Uncertainty", html_args...)
    add_markers!(p2_html, obs_easting, obs_northing, hidden_easting, hidden_northing)

    p3_html = contourf(x_coords, y_coords, score_matrix; levels=20, color=:jet, title="(C) AI Target Score", html_args...)
    add_markers!(p3_html, obs_easting, obs_northing, hidden_easting, hidden_northing)
    add_ai_targets!(p3_html, batch_rec_E, batch_rec_N, pomcp_rec_E, pomcp_rec_N)

    p4_html = surface(x_coords, y_coords, mean_matrix; color=:terrain, camera=(35, 45), title="(D) 3D Grade Topography", zlabel="Au Grade", html_args...)

    html_dashboard = plot(p1_html, p2_html, p3_html, p4_html, layout=(2, 2), size=(1400, 1000), margin=10Plots.mm)
    savefig(html_dashboard, joinpath(folder_name, "Interactive_$(file_name).html"))

    # ==========================================
    # 5. 生成完整TXT报告（已补全备选靶区）
    # ==========================================
    println("=== 5. 正在生成推荐报告 ===")
    report_path = joinpath(folder_name, "Report_$(file_name).txt")
    open(report_path, "w") do io
        write(io, "=================================================\n")
        write(io, "        AI 智能矿产勘探推荐报告\n")
        write(io, "        数据文件: $file_name\n")
        write(io, "        生成时间: $(Dates.now())\n")
        write(io, "=================================================\n\n")
        
        write(io, "【数据使用情况】\n")
        write(io, "  输入总钻孔数: $(n_total)\n")
        write(io, "  盲测保留比例: $(keep_ratio * 100)%\n")
        write(io, "  AI决策使用孔数: $(length(obs_quals))\n")
        write(io, "  隐藏验证孔数: $(length(hidden_quals))\n")
        write(io, "  运算网格大小: $(grid_nx) × $(grid_ny)\n")
        write(io, "  网格分辨率: $(round(current_cell_size, digits=2)) 米/格\n\n")

        write(io, "【1. 绝对最优靶区 (POMCP 单步推演)】\n")
        if pomcp_text != ""
            write(io, pomcp_text * "\n\n")
        else
            write(io, "当前无需继续打孔，建议开采或放弃。\n\n")
        end
        
        write(io, "【2. 批量备选靶区 (UCB 算法, 保持安全孔距)】\n")
        write(io, "  共筛选出 $(length(batch_rec_E)) 个满足最小间距要求的靶区\n")
        for i in 1:length(batch_rec_E)
            write(io, "  靶区 #$i -> Easting: $(round(batch_rec_E[i], digits=2)), Northing: $(round(batch_rec_N[i], digits=2))\n")
        end
        
        write(io, "\n=================================================\n")
        write(io, "注: 请结合 HighRes_$(file_name).png (静态) 或 Interactive_$(file_name).html (动态) 查看具体落点。\n")
        write(io, "图例提示: 红星=已知孔, 白圈=隐藏验证孔, 绿星=POMCP最优靶区, 黑叉=UCB备选靶区。\n")
    end

    println("✅ 完成处理: $file_name")
end

println("\n🏆 全部文件批量处理完毕！请在输出目录查看结果。")