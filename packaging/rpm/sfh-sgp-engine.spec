%global debug_package %{nil}

Name:           sfh-sgp-engine
Version:        1.0.0
Release:        1%{?dist}
Summary:        SFH-SGP Platform — Signal Geometry Engine V2_079

License:        Apache 2.0
URL:            https://github.com/urmt/SFH-SGP-Platform
Source0:        sfh-sgp-engine-%{version}.tar.gz

BuildRequires:  python3-devel
BuildRequires:  python3-pip
BuildRequires:  python3-setuptools

Requires:       python3-numpy
Requires:       python3-scipy
Requires:       python3-scikit-learn
Requires:       python3-fastapi
Requires:       python3-uvicorn
Requires:       python3-pydantic

%description
The SFH-SGP Platform provides tools for time-series analysis via canonical
four-dimensional embeddings under the frozen V2_079 architecture.
Includes a Python computation engine, CLI, and FastAPI REST+WebSocket server.

%prep
%autosetup -n sfh-sgp-engine-%{version}

%build
cd engine
%{python3} -m build --wheel --no-isolation

%install
cd engine
%{python3} -m pip install --force-reinstall --no-deps \
  --prefix=%{_prefix} --root=%{buildroot} \
  dist/sfh_sgp_engine-*.whl
# Install data files
install -d %{buildroot}%{_datadir}/sfh-sgp/metadata
install -m 644 metadata/V2_079_METADATA.json %{buildroot}%{_datadir}/sfh-sgp/metadata/
install -d %{buildroot}%{_datadir}/sfh-sgp/theory
for f in SFH_SGP_METADATA.json SFH_SGP_CANONICAL_THEORY.json \
         SFH_SGP_AXIOMATIC_THEORY.json SFH_SGP_PROOF_TRACK.json; do
  install -m 644 ../"$f" %{buildroot}%{_datadir}/sfh-sgp/theory/
done
install -d %{buildroot}%{_docdir}/sfh-sgp-engine-%{version}
install -m 644 ../SFH_SGP_Systems_Analysis_App_Design.docx %{buildroot}%{_docdir}/sfh-sgp-engine-%{version}/
install -m 644 ../README.md %{buildroot}%{_docdir}/sfh-sgp-engine-%{version}/
cp -r ../examples %{buildroot}%{_docdir}/sfh-sgp-engine-%{version}/
rm -rf %{buildroot}%{_docdir}/sfh-sgp-engine-%{version}/examples/__pycache__

%check
cd engine
%{python3} -m pytest tests/ -v

%files
%{python3_sitelib}/sfh_sgp/
%{python3_sitelib}/*sfh_sgp_engine*.dist-info/
%{_bindir}/sfh-sgp
%{_datadir}/sfh-sgp/
%doc %{_docdir}/sfh-sgp-engine-%{version}/

%changelog
* Tue May 26 2026  - 1.0.0-1
- Initial RPM release of SFH-SGP Engine V2_079 (frozen)