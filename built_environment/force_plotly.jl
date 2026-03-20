filepath = joinpath("scripts", "solve_pomcpow.jl")
content = read(filepath, String)

# 1. 恢复被注释的画图代码 (如果你之前运行了 remove_plots.jl)
content = replace(content, "# using Plots" => "using Plots")
content = replace(content, "# fig = heatmap" => "fig = heatmap")
content = replace(content, "# fig = plot" => "fig = plot")
content = replace(content, "# display(fig)" => "display(fig)")

# 2. 清理上一次可能失败的 plotlyjs() 尝试
content = replace(content, "using Plots\nplotlyjs()" => "using Plots")

# 3. 终极杀招：在 using Plots 之前注入环境变量，从根本上禁用 GR
if !occursin("PLOTS_DEFAULT_BACKEND", content)
    injection = """
    # 强制禁用 GR 后端，使用纯浏览器渲染的 Plotly
    ENV["PLOTS_DEFAULT_BACKEND"] = "Plotly"
    using Plots
    plotly()
    """
    content = replace(content, "using Plots" => injection)
end

write(filepath, content)
println("✓ 已强制在脚本开头注入 ENV[\"PLOTS_DEFAULT_BACKEND\"] = \"Plotly\"")
println("✓ GR 后端已被彻底屏蔽！")