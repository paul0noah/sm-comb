# SM-Comb

This repository contains the code of the paper 
"A Scalable Combinatorial Solver for Elastic Geometrically Consistent 3D Shape Matching", P. Roetzer, P. Swoboda, D. Cremers, F. Bernard. IEEE Conference on Computer Vision and Pattern Recognition (CVPR) 2022.
It aims to solve the Integer Linear Program for Shape Matching (ILP-SM) proposed by Windheuser et al. in "Geometrically Consistent Elastic Matching of 3D Shapes: A Linear Programming Solution".


## ⚙️ Installation

### 🐍 Python
Simply download the repo and `python setup.py install` (you need working c++ compiler + cmake installed on your machine).

### 🚀 Cpp

Clone the project and run the following commands inside the projects `c++` folder:
```
git submodule update --init --recursive
cd c++
./BUILD.sh
```
If you want the XCODE IDE run: 
```
./GENXCODEPROJECT.sh
```
The build currently is only tested on MacOS. If you have difficulties building the project contact `paul.roetzer@tum.de`.

### 🚧 Troubleshooting
#### OpenMP not found on MacOS
    On MacOS we experienced the problem problems of cmake finding OpenMP library.
    - Try installing OpenMP with Homebrew `brew install libomp`
    - Add the following flags to the cmake command in either `BUILD.sh` or `GENXCODEPROJECT.sh`
    ```
        -GXcode -DOpenMP_CXX_FLAGS="-Xclang -fopenmp -I /path/to/libomp/<version>/include/" -DOpenMP_C_FLAGS="-Xclang -fopenmp -I /path/to/libomp/<version>/include/" -DOpenMP_CXX_LIB_NAMES=libomp  -DOpenMP_C_LIB_NAMES=libomp -DOpenMP_libomp_LIBRARY=//path/to/libomp/<version>/lib/libomp.dylib -DCMAKE_SHARED_LINKER_FLAGS="-L /path/to/libomp/<version>/lib -lomp -Wl,-rpath, /path/to/libomp/<version>/lib"
    ```
    Usually `path/to/libomp` is `/opt/homebrew/Cellar/libomp/`

## ✨ Usage

### 🐍 Python
Vanilla usage as used in our paper:

```python
from shape_match_model_pb import ShapeMatchModel
import numpy as np
from scipy import sparse

# create smm model in which shapes X and Y (.ply or .off files) are reduced to 100 and 110 triangles respectively
# (if each shape contains already less triangles, the shapes are not reduced)
# Note: size of ILP grows quadratically with number of triangles of one of the shape 
# We recommend using less than 1000 triangles for each shape
smm = ShapeMatchModel("yourFile1{.ply, .off}", 100, "yourFile2{.ply, .off}", 110)

# solve the LP (with our combinatorial solver) and obtain binary vector G
G = smm.solve()

# check if solution fullfills constraints
if not smm.constraintsFullfilled():
    print("SM-comb could NOT find primal feasible solution")

# convert G to sparse int8 matrix
sG = sparse.csr_matrix(G, dtype=np.int8)

# get point correspondences from solution
p_c = smm.getPointMatchesFromSolution(sG)

# plot result color coded
smm.plotSolution(sG)
``` 

Other available function:
```python
# smm can be created also by passing numpy arrays FX contains triangles of shape X and VX contains vertex coordinates of shape X
smm = smm = ShapeMatchModel(FX, VX, FY, VY)

# check if smm was created successfully (i.e. if shapes are watertight)
if smm.smmCreatedSuccessFully():
    print("smm created successfully")
    
# update the energy function by using a |VX| x |VY| sized feature difference matrix
FEAT_DIFF = .. # np array of size |VX| x |VY| where lower values mean more similar features
smm.updateEnergy(FEAT_DIFF, True, False, 0) 
 
# export as lp file
smm.saveAsLPFile("your_lp_filename.lp")

# return final energy getFinalEnergy (note this is only the upperbound if `smm.constraintsFullfilled() == True`)
energy = smm.getFinalEnergy(sG)

# extract the lower bound
lb = smm.getLowerBound()

# return triangle-based matchings
fx_sol = smm.getFXCombo()[sG.indices, :]
fy_sol = smm.getFYCombo()[sG.indices, :]

# return ilp object (readable by bdd solver)
ilp = smm.getIlpObj()

```


### 🚀 Cpp

#### Requirements: 
- `OpenMP` not mandatory but highly recommended
- `OpenGL` for plotting results

#### Usage: 
To perfrom shape matching you will need to create a so-called `ShapeMatchModel` object. This object provides the API necessary to solve the ILP-SM, write the ILP-SM to a file, plot results and write the model to file.
##### Creating a ShapeMatchModel: 
The following example code creates a `ShapeMatchModel` object with the modelname `myShapeMatchModel`:
```c++
#include "shapeMatchModel/shapeMatchModel.hpp"
int main(int argc, char *argv[]) {
    const std::string filenameShape1 = "path/to/your/shape1.{.ply, .off}";
    const std::string filenameShape2 = "path/to/your/shape2.{.ply, .off}";
    ShapeMatchModelOpts opts;
    opts.modelName = "myShapeMatchModel";
    ShapeMatchModel smm(filenameShape1, filenameShape2, opts);
}
```
Note: your shapes have to closed and oriented triangle meshes. If they aren't you e.g. can use Meshalb to make the mesh oriented (`Filters > Normals, Curvature, ... > Re-Orient all faces coherently`) as well as closed (`Filters > Remeshing, ... > Close Holes`)

You can solve the ILP-SM as well as plot the solution as follows:
```c++
MatrixInt8 Gamma = smm.solve();
smm.plotInterpolatedSolution(Gamma);
```
##### Exporting Model:
There are several ways to export the mode
- Exporting Model as `.lp` with `smm.saveIlpAsLp(const std::string& filename)`
- Exporting Model `smm.writeModelToFile()` (assumes you provided a modelname)
Writing the model to file creates several files:
`<modelname>_X.ply`: shape X
`<modelname>_Y.ply`: shape Y
`<modelname>_E.csv`: the energy vector
`<modelname>_NWMH.csv`: internal information for the shape match model object
When `opts.writeModelToFileAfterSolving == true` additional files are written after solving:
`<modelname>_Gamma.csv`: the vector containing the solution (triangle correspondences)
`<modelname>_time.csv`: contains the durations `total time, time heuristic, time dual solver`

#### Partial Shapes
Partial Shapes are a special case and a little more time demanding to set up. We explain a potential workflow:
- choose a shape of the Tosca partial dataset
- load the shape into meshlab, make sure it is oriented and export the shape as `<shapename1>original.ply`
- close the holes of the shape and export its as `<shapename1>.ply`
- follow the same procedure for the other shape. If the other shape does not contain holes then export the shape as `<shapename2>.ply`

After setting everything use the following code to create a shape match model while providing the target number of faces you want to perform the matching on.
Note: with this approach you are able to produce plots of the matched models which contain holes.
```c++
#include "partialShapesHandler.hpp"
int main(int argc, char *argv[]) {
    const std::string filenameShape1 = "path/to/your/<shapename1>.ply";
    const bool didShape1ContainHoles = {true, false};
    const std::string filenameShape2 = "path/to/your/<shapename2>.ply";
    const bool didShape2ContainHoles = {true, false};
    PartialShapesHandler psh(filenameShape1", didShape1ContainHoles, filenameShape2, didShape2ContainHoles);
    ShapeMatchModelOpts opts;
    opts.modelName = "name/of/my/awesome/partialshapes/experiment/";
    opts.writeModelToFileAfterSolving = true;
    const int numFacesForShapeMatching = 500;
    ShapeMatchModel model = psh.generateShapeMatchoModel(opts, numFacesForShapeMatching);
    ...
}


    int numFaces = 600;
    ShapeMatchModelOpts opts;
    opts.modelName = "../experiments/partial/michael/victoria4_A0.90_H50_victoria12_A0.40_H5_" + std::to_string(numFaces);
    opts.writeModelToFileAfterSolving = true;
```


##### Available Options: 
ShapeMatchModelOpts:
```c++
    // toggle output of ShapeMatchModel
    opts.verbose; 
    // limit the number of dual solver calls (might result in unfinished primal solution)
    opts.maxNumDualSolverCalls; 
    // if false, dual problem will not be constructed and solved
    opts.useMinMarginals; 
    // name of the shape match model and of the corresonding files
    opts.modelName; 
    // if true, the energy, the solution Gamma and both shapes get written to files (see below)
    opts.writeModelToFileAfterSolving; 
    // options for the primal heuristic (see below)
    opts.primalHeuristicOpts; 
    // options for bdd solver (see below)
    opts.bddSolverOpts; 
```
PrimalHeuristicOpts:
```c++
    // max iterations primal heuristic runs
    opts.primalHeuristicOpts.maxIter; 
    // primal heuristic stops if number of backtracks is larger than maxBacktracks 
    opts.primalHeuristicOpts.maxBacktracks; 
    // toggle output of primal heuristic
    opts.primalHeuristicOptsverbose; 
    // use to round tri-tri matchings only
    opts.primalHeuristicOptsallowOnlyNonDegenerateMatchings; 
    // if you choose your own maxIter set this to false
    opts.primalHeuristicOptsautoSetMaxBacktracks; 
    // if you choose your own maxBacktracks set this to false
    opts.primalHeuristicOptsautoSetMaxIter; 
    // default is 100, you may reduce this for larger shapes to obtain faster solution (with higher risk of wrong initialization)
    opts.primalHeuristicOptsmaxNumInitCandidates; 
```
Dual Solver options
```c++
    // choose implementation of implementation = {parallel_mma, sequential_mma}
    opts.bddSolverOpts.bdd_solver_impl_ = LPMP::bdd_solver_options::bdd_solver_impl::implementation;
    // precision of solver
    bddSolverOpts.bdd_solver_precision_ = LPMP::bdd_solver_options::bdd_solver_precision::single_prec;
    // tolerance for stopping criterion (relative change to previous lower bound smaller than tolerance)
    bddSolverOpts.tolerance = 1e-6;
```

## 🔡 Misc

### References 
- [1] Windheuser, T., Schlickwei, U., Schimdt, F. R., & Cremers, D. (2011, August). Large‐scale integer linear programming for orientation preserving 3d shape matching. In Computer Graphics Forum (Vol. 30, No. 5, pp. 1471-1480). Oxford, UK: Blackwell Publishing Ltd.  
- [2] Windheuser, T., Schlickewei, U., Schmidt, F. R., & Cremers, D. (2011, November). Geometrically consistent elastic matching of 3d shapes: A linear programming solution. In 2011 International Conference on Computer Vision (pp. 2134-2141). IEEE.  
- [3] Lange, J. H., & Swoboda, P. (2021, July). Efficient Message Passing for 0–1 ILPs with Binary Decision Diagrams. In International Conference on Machine Learning (pp. 6000-6010). PMLR.  
- [4] P. Roetzer, P. Swoboda, D. Cremers, F. Bernard (2021). A Scalable Combinatorial Solver for Elastic Geometrically Consistent 3D Shape Matching. In IEEE Conference on Computer Vision and Pattern Recognition (CVPR). 2022

### Used Libraries
| Lib Name | Functionality |
| ------ | ------ |
| [BDD-Solver](https://github.com/LPMP/BDD) | Solves Dual Problem Efficiently |
| [libigl](https://github.com/libigl/libigl) | Handles everything 3D Shapes related  | 
| [Eigen3](https://gitlab.com/libeigen/eigen) | Convenient Matrix/Vector Handling |
| [Robin-Map](https://github.com/Tessil/robin-map) | Hash Maps, Hash Sets |

## 🎓 Attribution 
```bibtex
@inproceedings{roetzer2022scalable,
  title={A scalable combinatorial solver for elastic geometrically consistent 3d shape matching},
  author={Roetzer, Paul and Swoboda, Paul and Cremers, Daniel and Bernard, Florian},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={428--438},
  year={2022}
}
```
