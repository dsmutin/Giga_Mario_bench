/*
 * Run start-to-end cells written by `giga_mario_bench score prepare`.
 * Each Python script checks which artifacts already exist and skips them.
 */
nextflow.enable.dsl = 2

params.exec_root = "src/giga_mario_bench/exec"
params.python = "python"

workflow {
    scripts = Channel
        .fromPath("${params.exec_root}/**/*.py")
        .filter { path -> path.name != "__init__.py" && path.name != "manifest.json" }
    scripts | run_cell
}

process run_cell {
    tag { script.baseName }
    debug true

    input:
    path script

    script:
    """
    ${params.python} ${script}
    """
}
