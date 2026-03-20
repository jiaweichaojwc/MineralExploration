using Pkg

# 1. 自动修复 DelimitedFiles 依赖警告
project_file = "Project.toml"
project_content = read(project_file, String)
if !occursin("DelimitedFiles", project_content)
    # 找到 [deps] 并在下方插入 DelimitedFiles 的 UUID
    project_content = replace(project_content, "[deps]" => "[deps]\nDelimitedFiles = \"8bb1440f-4735-579b-a4ab-409b98df4dab\"")
    write(project_file, project_content)
    println("✓ 已将标准库 DelimitedFiles 补充声明至 Project.toml")
end

# 2. 自动修复 POMDPs.jl 0.9 版本的 API 变更
function patch_file(filepath)
    if isfile(filepath)
        content = read(filepath, String)
        if occursin("initialstate_distribution", content)
            # 批量将旧 API 替换为新 API
            new_content = replace(content, "initialstate_distribution" => "initialstate")
            write(filepath, new_content)
            println("✓ 已自动修复源码文件中的 API: ", filepath)
        end
    end
end

# 修复核心源文件
patch_file(joinpath("src", "pomdp.jl"))
patch_file(joinpath("src", "MineralExploration.jl"))
patch_file("README.md")

# 修复所有的执行脚本 (使得用户后续运行 solve_pomcpow.jl 等脚本时也不会报错)
scripts_dir = "scripts"
if isdir(scripts_dir)
    for file in readdir(scripts_dir)
        if endswith(file, ".jl")
            patch_file(joinpath(scripts_dir, file))
        end
    end
end

println("\n==================================================")
println("代码级 API 兼容性修复完成！正在执行最后一次预编译...")
println("==================================================")

# 激活环境并重新预编译
Pkg.activate(".")
Pkg.resolve()
Pkg.instantiate()
Pkg.precompile()

println("恭喜，环境和代码已全部完美修复，可以开始运行你的地质探索项目了！")