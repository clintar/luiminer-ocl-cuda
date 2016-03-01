
#  JANSSON_FOUND - System has Jansson
#  JANSSON_INCLUDE_DIRS - The Jansson include directories
#  JANSSON_LIBRARIES - The libraries needed to use Jansson
#  JANSSON_DEFINITIONS - Compiler switches required for using Jansson

find_path(JANSSON_INCLUDE_DIR NAMES jansson.h
	PATHS
	/usr/local/include
  	/usr/include
  	"C:/Program Files (x86)/Microsoft Visual Studio 11.0/VC/include"
)

find_library(JANSSON_LIBRARY NAMES 
	jansson
	jansson.x86
	PATHS
    /usr/local/lib
    /usr/lib
    "C:/Program Files (x86)/Microsoft Visual Studio 11.0/VC/lib"
)

set(JANSSON_LIBRARIES ${JANSSON_LIBRARY} )
set(JANSSON_INCLUDE_DIRS ${JANSSON_INCLUDE_DIR} )

include(FindPackageHandleStandardArgs)

# handle the QUIETLY and REQUIRED arguments and set JANSSON_FOUND to TRUE
# if all listed variables are TRUE

find_package_handle_standard_args(Jansson  DEFAULT_MSG
                                  JANSSON_LIBRARY JANSSON_INCLUDE_DIR)
mark_as_advanced(JANSSON_INCLUDE_DIR JANSSON_LIBRARY )


if(JANSSON_FOUND)
  set(JANSSON_LIBRARIES ${JANSSON_LIBRARY})
  set(JANSSON_INCLUDE_DIRS ${JANSSON_INCLUDE_DIR})
endif()