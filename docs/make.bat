@ECHO OFF
pushd %~dp0

if "%SPHINXBUILD%" == "" set SPHINXBUILD=sphinx-build
%SPHINXBUILD% -W --keep-going -b html . _build/html %*

popd
