"""Portable frozen ERFA/WGS84 illumination functions; no external pipeline import."""
import numpy as np
import erfa
AXES_KM = np.array([6378.137,6378.137,6378.137*(1-1/298.257223563)])

def circular_median(values, period):
    values = np.asarray(values, dtype=float)
    a = values*2*np.pi/period
    center = np.arctan2(np.sin(a).mean(), np.cos(a).mean())*period/(2*np.pi)
    return (center + np.median((values-center+period/2) % period-period/2)) % period


def sun_ecef(sec):
    # Original ERFA model: geometric Sun, TT~TDB, UT1~UTC, polar motion zero.
    u1 = 2440587.5 + np.floor(sec/86400)
    u2 = (sec % 86400)/86400
    ta1, ta2 = erfa.utctai(u1, u2)
    tt1, tt2 = erfa.taitt(ta1, ta2)
    earth_helio, _ = erfa.epv00(tt1, tt2)
    icrs = -earth_helio['p']*erfa.DAU/1000
    rot = erfa.c2t06a(tt1, tt2, u1, u2, 0., 0.)
    return np.einsum('nij,nj->ni', rot, icrs)


def ray_blocked(p, u):
    p, u = p/AXES_KM, u/AXES_KM
    a = np.sum(u*u, axis=1)
    b = np.sum(p*u, axis=1)
    c = np.sum(p*p, axis=1)-1
    return (c < 0) | ((b < 0) & (b*b-a*c >= 0))


def solar_fields(utc_s, lat_deg, lon_deg, height_km, xyz_km=None):
    """Solar geometry at actual WGS84 reference points; no sampling/quality mask."""
    utc_s, lat_deg, lon_deg, height_km = [np.asarray(a, dtype=float) for a in
                                       [utc_s, lat_deg, lon_deg, height_km]]
    assert all(a.shape == utc_s.shape for a in [lat_deg, lon_deg, height_km])
    assert np.isfinite(np.c_[utc_s, lat_deg, lon_deg, height_km]).all()
    unique, inv = np.unique(utc_s, return_inverse=True)
    solar = np.empty((len(unique), 3))
    for start in range(0, len(unique), 10000):
        solar[start:start+10000] = sun_ecef(unique[start:start+10000])
    sun = solar[inv]
    lat, lon = np.deg2rad(lat_deg), np.deg2rad(lon_deg)
    reconstructed = erfa.gd2gc(1, lon, lat, height_km*1000)/1000
    if xyz_km is None:
        xyz_km = reconstructed
    else:
        xyz_km = np.asarray(xyz_km, dtype=float)
        assert np.isfinite(xyz_km).all()
        assert np.max(np.linalg.norm(reconstructed-xyz_km, axis=1)) < 1e-4
    direction = sun-xyz_km
    direction /= np.linalg.norm(direction, axis=1)[:, None]
    up = np.c_[np.cos(lat)*np.cos(lon), np.cos(lat)*np.sin(lon), np.sin(lat)]
    sza = np.degrees(np.arccos(np.clip(np.sum(up*direction, axis=1), -1, 1)))
    sun_lon = np.degrees(np.arctan2(sun[:, 1], sun[:, 0]))
    sun_lat_geocentric = np.degrees(np.arctan2(sun[:, 2], np.hypot(sun[:, 0], sun[:, 1])))
    blocked = ray_blocked(xyz_km, direction)
    assert not np.any(blocked & (sza < 89.999))
    return {
        'solar_zenith_deg': sza,
        'solar_local_time_h': (12+(lon_deg-sun_lon)/15)%24,
        'sunlit': ~blocked,
        'sun_ecef_longitude_deg': sun_lon,
        'sun_ecef_latitude_geocentric_deg': sun_lat_geocentric,
    }
