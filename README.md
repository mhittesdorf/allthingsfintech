Source code for the Introducing QuantLib series published on my 'All things finance and technology' blog (http://mhittesdorf.wordpress.com)

Includes a CMakeLists.txt file to build the tests using cmake.  

Instructions to build on Linux (Ubuntu 12.04):

* Make sure the Boost (1.51) and QuantLib (1.21) libraries are installed on your machine. Header files should be in /usr/local/include. Shared libraries should be in /usr/local/lib.  These are the default installation directories.
* Create a working directory and clone the allthingsfintech project into your own git repository: git clone https://github.com/mhittesdorf/allthingsfintech.git
* Create a 'build' subdirectory under 'allthingsfintech' and 'cd' to it: cd allthingsfintech; mkdir build; cd build
* Make sure cmake is installed (sudo apt-get install cmake) and then generate makefile: cmake ..
* Build the application: make
* The result should be an executable called 'allthingsfintech'.  Run it with the command: ./allthingsfintech

NOTE: Though I've only tested this build on Linux, the instructions should be similar for Windows as cmake is a cross-platform build system.


Enjoy! 

- Mick Hittesdorf
