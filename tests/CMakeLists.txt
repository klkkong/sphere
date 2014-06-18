find_package(PythonInterp REQUIRED)

add_test(io_tests ${PYTHON_EXECUTABLE} 
    ${CMAKE_CURRENT_BINARY_DIR}/io_tests.py)

add_test(wall_contact_model_tests ${PYTHON_EXECUTABLE} 
    ${CMAKE_CURRENT_BINARY_DIR}/contactmodel_wall.py)

add_test(io_tests_fluid ${PYTHON_EXECUTABLE} 
    ${CMAKE_CURRENT_BINARY_DIR}/io_tests_fluid.py)

add_test(porosity_tests ${PYTHON_EXECUTABLE} 
    ${CMAKE_CURRENT_BINARY_DIR}/porosity_tests.py)

add_test(memory_tests ${PYTHON_EXECUTABLE} 
    ${CMAKE_CURRENT_BINARY_DIR}/memcheck_tests.py)

add_test(bond_tests ${PYTHON_EXECUTABLE} 
    ${CMAKE_CURRENT_BINARY_DIR}/bond_tests.py)

add_test(cfd_tests ${PYTHON_EXECUTABLE} 
    ${CMAKE_CURRENT_BINARY_DIR}/cfd_tests.py)

add_test(cfd_tests_neumann ${PYTHON_EXECUTABLE} 
    ${CMAKE_CURRENT_BINARY_DIR}/cfd_tests_neumann.py)

add_test(cfd_dirichlet_homogeneous ${PYTHON_EXECUTABLE} 
    ${CMAKE_CURRENT_BINARY_DIR}/cfd_dirichlet_homogeneous.py)
