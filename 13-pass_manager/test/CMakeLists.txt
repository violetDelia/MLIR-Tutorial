# 配置lit套件设置
configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
  MAIN_CONFIG
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
)
# 
set(TEST_DEPENDS
  FileCheck count not
  NS-opt${ch_num}
)

add_lit_testsuite(check-ch-${ch_num} "Running the lit regression tests..."
  ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS ${TEST_DEPENDS}
)
set_target_properties(check-ch-${ch_num} PROPERTIES FOLDER "test")

