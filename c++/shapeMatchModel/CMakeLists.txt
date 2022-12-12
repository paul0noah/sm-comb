add_subdirectory(constraintsComputation)
add_subdirectory(combinations)
add_subdirectory(energyComputation)

target_sources(${PROJECT_NAME} 
				PRIVATE
					shapeMatchModel.cpp
					shapeMatchModel.hpp
					primalHeuristic.cpp
					primalHeuristicInitialization.cpp
					primalHeuristic.hpp
					plotSolution.cpp
	)

if( BUILD_PYTHON_BINDINGS )
	target_sources(${PROJECT_NAME} PRIVATE shapeMatchModelPB.cpp)
endif()