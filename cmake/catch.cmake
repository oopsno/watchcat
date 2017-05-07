function(download_catch)
  set(URL https://github.com/philsquared/Catch/releases/download/v${CATCH_VERSION}/catch.hpp)
  set(TARGET ${PROJECT_INCLUDE_DIR}/catch.hpp)
  message(STATUS "Downloading ${URL}")
  file(DOWNLOAD ${URL} ${TARGET}
    EXPECTED_HASH SHA224=${CATCH_SHA224}
    SHOW_PROGRESS)
endfunction()
