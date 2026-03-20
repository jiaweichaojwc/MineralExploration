filepath = joinpath("scripts", "solve_pomcpow.jl")
if !isfile(filepath)
    println("ERROR: 找不到 scripts/solve_pomcpow.jl 文件。")
    exit(1)
end

println("正在读取并准备修改 scripts/solve_pomcpow.jl...")
content = read(filepath, String)

# === 步骤 1：撤销之前的禁用操作，恢复画图核心代码 ===

# 恢复 `using Plots`，但同时指定使用 plotlyjs 后端
content = replace(content, r"(#\s*)using Plots" => "using Plots\nplotlyjs()") # 放开并设置后端

# 恢复画热力图
content = replace(content, r"(#\s*)fig = heatmap" => "fig = heatmap")

# 恢复画信念更新图
content = replace(content, r"(#\s*)fig = plot" => "fig = plot")

# === 步骤 2：彻底恢复显示功能 (撤销更早前的display fig注释) ===

# 这一行之前被建议注释掉躲避GR弹窗报错，现在必须放开，PlotlyJS会弹出浏览器窗口
content = replace(content, r"(#\s*)display\(fig\)" => "display(fig)")


# === 步骤 3：确保不加载那几个不兼容的性能工具 (保持现状) ===

# 这里确保它们还是被注释掉的状态，因为你的环境还是编译不过它们
if !occursin("# using ProfileView", content) && occursin("using ProfileView", content)
    content = replace(content, "using ProfileView" => "# using ProfileView")
end
if !occursin("# using D3Trees", content) && occursin("using D3Trees", content)
    content = replace(content, "using D3Trees" => "# using D3Trees")
end


write(filepath, content)
println("\n✓ 已完成 scripts/solve_pomcpow.jl 的修改：")
println("   1. 绘图代码已恢复。")
println("   2. 绘图后端已强制切换为 PlotlyJS (使用浏览器弹图)。")
println("   3. 注释了 display(fig) 行已恢复。")
println("   4. 性能工具 ProfileView/D3Trees 保持禁用状态。")
println("\n现在请在命令行运行主模拟脚本。")