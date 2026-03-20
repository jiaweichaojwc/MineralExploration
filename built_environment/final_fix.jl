using Pkg

Pkg.activate(".")

# 删除残留的清单文件以确保完全遵守新的 Project.toml 约束
if isfile("Manifest.toml")
    rm("Manifest.toml")
    println("已删除旧的 Manifest.toml")
end

println("正在重新解析兼容的包版本...")
Pkg.resolve()
Pkg.instantiate()

println("==================================================")
println("正在执行最终的预编译过程，这次应该能全部通过：")
println("==================================================")
Pkg.precompile()

println("环境构建完毕！")