CMAKE_MINIMUM_REQUIRED(VERSION 3.11)

#Optimization with system calls
ADD_EXECUTABLE(branin_system_calls ./branin_system_calls.cpp)
add_dependencies(branin_system_calls bayesopt)
TARGET_LINK_LIBRARIES(branin_system_calls bayesopt)

#Optimization with xml and system calls
ADD_EXECUTABLE(branin_xml ./branin_xml.cpp)
add_dependencies(branin_xml bayesopt)
TARGET_LINK_LIBRARIES(branin_xml bayesopt)

#Continuous test
ADD_EXECUTABLE(bo_cont ./bo_cont.cpp)
add_dependencies(bo_cont bayesopt)
TARGET_LINK_LIBRARIES(bo_cont bayesopt)

#Discrete test
ADD_EXECUTABLE(bo_disc ./bo_disc.cpp)
add_dependencies(bo_disc bayesopt)
TARGET_LINK_LIBRARIES(bo_disc bayesopt)

#1D test
ADD_EXECUTABLE(bo_oned ./bo_oned.cpp)
add_dependencies(bo_oned bayesopt)
TARGET_LINK_LIBRARIES(bo_oned bayesopt)

#Branin
ADD_EXECUTABLE(bo_branin ./bo_branin.cpp )
add_dependencies(bo_branin bayesopt)
TARGET_LINK_LIBRARIES(bo_branin bayesopt)

ADD_EXECUTABLE(bo_branin_timed ./bo_branin_timed.cpp )
add_dependencies(bo_branin_timed bayesopt)
TARGET_LINK_LIBRARIES(bo_branin_timed bayesopt)

ADD_EXECUTABLE(bo_branin_mcmc ./bo_branin_mcmc.cpp )
add_dependencies(bo_branin_mcmc bayesopt)
TARGET_LINK_LIBRARIES(bo_branin_mcmc bayesopt)

#Hartmann
ADD_EXECUTABLE(bo_hartmann ./bo_hartmann.cpp )
add_dependencies(bo_hartmann bayesopt)
TARGET_LINK_LIBRARIES(bo_hartmann bayesopt)

#Camelback
ADD_EXECUTABLE(bo_camelback ./bo_camelback.cpp )
add_dependencies(bo_camelback bayesopt)
TARGET_LINK_LIBRARIES(bo_camelback bayesopt)

ADD_EXECUTABLE(bo_compare ./bo_compare.cpp )
add_dependencies(bo_compare bayesopt)
TARGET_LINK_LIBRARIES(bo_compare bayesopt)
