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
# 🌟 只需修改这里：文件夹路径 🌟
# ==========================================
input_folder = raw"C:\\Users\\Deep-Lei\\Desktop\\转化后格式"
output_base_dir = raw"C:\\Users\\Deep-Lei\\Desktop"

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
# 🔄 主循环：对每个 CSV 执行一遍你的原代码
# ==========================================
for csv_file_path in csv_files
    
    # --- 获取当前文件名 (用于生成独立文件夹) ---
    file_name = splitext(basename(csv_file_path))[1]
    println("\n🚀 正在处理: $file_name ...")

    # --- 创建独立的结果文件夹 ---
    timestamp = Dates.format(Dates.now(), "yyyymmdd_HHMMSS")
    folder_name = joinpath(output_base_dir, "Result_$(file_name)_$timestamp")
    mkpath(folder_name)

    # ==========================================
    # 🌟 以下是你原来的代码，几乎没改，只是把 input_csv_path 换成了循环变量
    # ==========================================

    # 1. 路径配置 (现在用循环里的 csv_file_path)
    input_csv_path = csv_file_path 
    
    # 2. 核心计算参数
    cell_size = 10       
    padding = 50.0         
    min_grid_dist = 3.0    

    # 3. 盲测参数
    keep_ratio = 1       
    random_seed = 42       

    # ==========================================
    # 0. 创建文件夹 (上面已经创建过了，这里保留备份逻辑)
    # ==========================================
    println("=== 📁 结果保存至: $folder_name ===")

    # ==========================================
    # 1. 读取数据 (这里开始完全是你的原逻辑)
    # ==========================================
    println("=== 1. 读取数据 ===")
    df = CSV.read(input_csv_path, DataFrame)
    cp(input_csv_path, joinpath(folder_name, "backup_$(file_name).csv"), force=true) # 备份文件名加上当前文件名
    n_total = nrow(df)

    # 盲测逻辑
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

    # 计算边界框
    min_E = minimum(df.X) - padding
    max_E = maximum(df.X) + padding
    min_N = minimum(df.Y) - padding
    max_N = maximum(df.Y) + padding
    span_E = max_E - min_E
    span_N = max_N - min_N

    # 防内存溢出
    max_grid_dim = 60 
    current_cell_size = cell_size 
    if (span_E / current_cell_size) > max_grid_dim || (span_N / current_cell_size) > max_grid_dim
        safe_cell_size = max(span_E / max_grid_dim, span_N / max_grid_dim)
        current_cell_size = safe_cell_size
    end
    grid_nx = max(20, ceil(Int, span_E / current_cell_size)) 
    grid_ny = max(20, ceil(Int, span_N / current_cell_size)) 

    # 映射坐标
    obs_coords = zeros(Int64, 2, length(obs_easting))
    for i in 1:length(obs_easting)
        norm_E = (obs_easting[i] - min_E) / span_E
        norm_N = (obs_northing[i] - min_N) / span_N
        obs_coords[1, i] = clamp(round(Int, norm_E * (grid_nx - 1)) + 1, 1, grid_nx)
        obs_coords[2, i] = clamp(round(Int, norm_N * (grid_ny - 1)) + 1, 1, grid_ny)
    end
    my_data = RockObservations(ore_quals=obs_quals, coordinates=obs_coords)

    # ==========================================
    # 2. 初始化模型
    # ==========================================
    auto_gp_mean = sum(obs_quals) / length(obs_quals)
    m = MineralExplorationPOMDP(
        reservoir_dims = (span_E, span_N, 30.0), 
        grid_dim = (grid_nx, grid_ny, 1),
        max_bores = 10, delta = 2,
        variogram = (0.005, 30.0, 0.0001), 
        gp_mean = auto_gp_mean, initial_data = my_data       
    )
    up = MEBeliefUpdater(m, 1000, 2.0)
    MineralExploration.update!(up.geostats, my_data)
    ds0 = POMDPs.initialstate(m)
    MineralExploration.update!(ds0.gp_distribution, my_data)
    b0 = POMDPs.initialize_belief(up, ds0)
    mean_ore, var_ore = MineralExploration.summarize(b0)

    # ==========================================
    # 3. 计算推荐坐标
    # ==========================================
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
    pomcp_text = ""
    if best_action.type == :drill
        rec_E = min_E + (best_action.coords[1] - 1) * (span_E / (grid_nx - 1))
        rec_N = min_N + (best_action.coords[2] - 1) * (span_N / (grid_ny - 1))
        push!(pomcp_rec_E, rec_E)
        push!(pomcp_rec_N, rec_N)
        pomcp_text = "✅ [POMCP 单步最优靶区]: (Easting: $(round(rec_E, digits=2)), Northing: $(round(rec_N, digits=2)))"
    end

    # UCB 批量
    candidates = []
    for x in 1:grid_nx, y in 1:grid_ny
        if !any((obs_coords[1, :] .== x) .& (obs_coords[2, :] .== y))
            score = mean_ore[x, y, 1] + 2.0 * sqrt(var_ore[x, y, 1])
            push!(candidates, (x=x, y=y, score=score))
        end
    end
    sort!(candidates, by = cand -> cand.score, rev=true)
    selected_points = []
    for cand in candidates
        if !any(sqrt((cand.x - sp.x)^2 + (cand.y - sp.y)^2) < min_grid_dist for sp in selected_points)
            push!(selected_points, cand)
            length(selected_points) >= 200 && break
        end 
    end
    batch_rec_E = [min_E + (p.x - 1) * (span_E / (grid_nx - 1)) for p in selected_points]
    batch_rec_N = [min_N + (p.y - 1) * (span_N / (grid_ny - 1)) for p in selected_points]

    # ==========================================
    # 4. 可视化 (保存文件名加上 file_name 防止覆盖)
    # ==========================================
    x_coords = range(min_E, max_E, length=grid_nx)
    y_coords = range(min_N, max_N, length=grid_ny)
    mean_matrix = mean_ore[:,:,1]' 
    std_matrix = sqrt.(var_ore[:,:,1])'
    score_matrix = mean_matrix .+ 2.0 .* std_matrix

    function add_markers!(p)
        scatter!(p, obs_easting, obs_northing, markershape=:star5, markersize=3.5, color=:red, markerstrokecolor=:black, markerstrokewidth=0.5, label="Observed")
        if length(hidden_easting) > 0
            scatter!(p, hidden_easting, hidden_northing, markershape=:circle, markersize=2.5, color=:white, markerstrokecolor=:black, markerstrokewidth=0.5, alpha=0.8, label="Hidden")
        end
    end
    function add_ai_targets!(p)
        if length(batch_rec_E) > 0
            scatter!(p, batch_rec_E, batch_rec_N, markershape=:xcross, markersize=7, color=:black, markerstrokewidth=2, label="Top 5 Targets")
        end
        if length(pomcp_rec_E) > 0
            scatter!(p, pomcp_rec_E, pomcp_rec_N, markershape=:star8, markersize=13, color=:lime, markerstrokecolor=:black, markerstrokewidth=1, label="Best POMCP")
        end
    end

    # 4.1 PNG
    gr(dpi=300)
    png_args = (xlabel="Easting (m)", ylabel="Northing (m)", tickfont=font(7, "Arial"), xrotation=45, titlefont=font(11, "Arial", :bold), guidefont=font(9, "Arial", :bold), linewidth=0, framestyle=:box, bottom_margin=12Plots.mm, left_margin=10Plots.mm, right_margin=22Plots.mm, colorbar_tickfont=font(7, "Arial"))
    
    p1_png = contourf(x_coords, y_coords, mean_matrix; levels=20, color=:terrain, title="(A) Predicted Au Grade (Mean)", png_args...)
    add_markers!(p1_png)
    p2_png = contourf(x_coords, y_coords, std_matrix; levels=20, color=:inferno, title="(B) Exploration Uncertainty", png_args...)
    add_markers!(p2_png)
    p3_png = contourf(x_coords, y_coords, score_matrix; levels=20, color=:jet, title="(C) AI Target Score (Blind Test)", png_args...)
    add_markers!(p3_png); add_ai_targets!(p3_png)
    p4_png = surface(x_coords, y_coords, mean_matrix; color=:terrain, camera=(35, 45), title="(D) 3D Grade Topography", zlabel="Au Grade", png_args...)
    
    png_dashboard = plot(p1_png, p2_png, p3_png, p4_png, layout=(2, 2), size=(1920, 1200), margin=18Plots.mm, background_color=:white)
    savefig(png_dashboard, joinpath(folder_name, "HighRes_$(file_name).png")) # 文件名修改

    # 4.2 HTML
    plotly()
    html_args = (xlabel="Easting (m)", ylabel="Northing (m)", tickfont=font(9), titlefont=font(12, :bold), linewidth=0)
    
    p1_html = contourf(x_coords, y_coords, mean_matrix; levels=20, color=:terrain, title="(A) Predicted Au Grade (Mean)", html_args...)
    add_markers!(p1_html)
    p2_html = contourf(x_coords, y_coords, std_matrix; levels=20, color=:inferno, title="(B) Exploration Uncertainty", html_args...)
    add_markers!(p2_html)
    p3_html = contourf(x_coords, y_coords, score_matrix; levels=20, color=:jet, title="(C) AI Target Score", html_args...)
    add_markers!(p3_html); add_ai_targets!(p3_html)
    p4_html = surface(x_coords, y_coords, mean_matrix; color=:terrain, camera=(35, 45), title="(D) 3D Grade Topography", zlabel="Au Grade", html_args...)
    
    html_dashboard = plot(p1_html, p2_html, p3_html, p4_html, layout=(2, 2), size=(1400, 1000), margin=10Plots.mm)
    savefig(html_dashboard, joinpath(folder_name, "Interactive_$(file_name).html")) # 文件名修改

    # ==========================================
    # 5. 生成报告
    # ==========================================
    report_path = joinpath(folder_name, "Report_$(file_name).txt")
    open(report_path, "w") do io
        write(io, "=================================================\n")
        write(io, "        AI 智能矿产勘探推荐报告\n")
        write(io, "        数据文件: $file_name\n")
        write(io, "=================================================\n\n")
        write(io, "【1. 绝对最优靶区】\n")
        if pomcp_text != ""
            write(io, pomcp_text * "\n")
        end
    end

    println("✅ 完成: $file_name")
end

println("\n🏆 全部处理完毕！")