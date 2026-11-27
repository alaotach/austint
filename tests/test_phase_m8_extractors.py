"""Offline tests for NormalizedProfile and 4-tier platform extractors."""

from __future__ import annotations

import httpx
import pytest
import respx

from osint_agent.collectors.extractors import (
    EXTRACTOR_REGISTRY,
    NormalizedProfile,
    ml_text_fallback_extract,
)


@pytest.mark.asyncio
@respx.mock
async def test_api_extractors_dockerhub_chess_codeforces_npm():
    respx.get("https://hub.docker.com/v2/users/alaotach/").mock(
        return_value=httpx.Response(
            200,
            json={
                "username": "alaotach",
                "full_name": "Aryan",
                "location": "Bengaluru",
                "date_joined": "2023-02-14T10:00:00Z",
                "gravatar_url": "https://gravatar.com/avatar/abc",
                "company": "OSS",
            },
        )
    )
    respx.get("https://hub.docker.com/v2/repositories/alaotach/").mock(
        return_value=httpx.Response(200, json={"count": 4})
    )
    respx.get("https://api.chess.com/pub/player/alaotach").mock(
        return_value=httpx.Response(
            200,
            json={
                "username": "alaotach",
                "name": "Aryan",
                "avatar": "https://images.chesscomfiles.com/a.jpg",
                "followers": 12,
                "joined": 1600000000,
                "last_online": 1700000000,
                "country": "https://api.chess.com/pub/country/IN",
            },
        )
    )
    respx.get("https://api.chess.com/pub/player/alaotach/stats").mock(
        return_value=httpx.Response(200, json={"chess_blitz": {"last": {"rating": 1450}}})
    )
    respx.get("https://codeforces.com/api/user.info").mock(
        return_value=httpx.Response(
            200,
            json={
                "status": "OK",
                "result": [
                    {
                        "handle": "alaotach",
                        "firstName": "Aryan",
                        "country": "India",
                        "rating": 1520,
                        "rank": "specialist",
                        "titlePhoto": "https://userpic.codeforces.org/1.jpg",
                        "registrationTimeSeconds": 1620000000,
                    }
                ],
            },
        )
    )
    respx.get("https://registry.npmjs.org/-/v1/search").mock(
        return_value=httpx.Response(
            200,
            json={
                "total": 2,
                "objects": [{"package": {"name": "eli5-cli", "links": {"npm": "https://www.npmjs.com/package/eli5-cli"}}}],
            },
        )
    )

    async with httpx.AsyncClient() as client:
        dh = await EXTRACTOR_REGISTRY["dockerhub"]("https://hub.docker.com/u/alaotach", "alaotach", client)
        ch = await EXTRACTOR_REGISTRY["chess"]("https://www.chess.com/member/alaotach", "alaotach", client)
        cf = await EXTRACTOR_REGISTRY["codeforces"]("https://codeforces.com/profile/alaotach", "alaotach", client)
        np = await EXTRACTOR_REGISTRY["npm"]("https://www.npmjs.com/~alaotach", "alaotach", client)

    assert dh.post_count == {"repos": 4}
    assert dh.joined_at == "2023-02-14"
    assert dh.field_sources["joined_at"] == "api"
    assert ch.followers == 12
    assert ch.extra["ratings"]["blitz"] == 1450
    assert cf.extra["rating"] == 1520
    assert np.post_count == {"packages": 2}


@pytest.mark.asyncio
@respx.mock
async def test_structured_extractors_soundcloud_youtube_telegram_steam_pypi_medium(monkeypatch):
    monkeypatch.delenv("YOUTUBE_API_KEY", raising=False)
    monkeypatch.delenv("STEAM_API_KEY", raising=False)

    respx.get("https://soundcloud.com/alaotach").mock(
        return_value=httpx.Response(
            200,
            text='<script>window.__sc_hydration = [{"hydratable":"user","data":{"username":"alaotach","full_name":"Aryan","followers_count":42,"track_count":5,"created_at":"2022-05-01T00:00:00Z","country_code":"IN","avatar_url":"https://i1.sndcdn.com/a.jpg"}}];</script>',
        )
    )
    respx.get("https://www.youtube.com/@alaotach").mock(
        return_value=httpx.Response(
            200,
            text='<meta property="og:title" content="alaotach"><meta property="og:description" content="Tech videos">"subscriberCountText":{"simpleText":"1.2K subscribers"}',
        )
    )
    respx.get("https://t.me/mrmango42").mock(
        return_value=httpx.Response(
            200,
            text='<div class="tgme_page_title">Mr Mango</div><div class="tgme_page_description">Channel bio</div><div class="tgme_page_extra">1 200 subscribers</div>',
        )
    )
    respx.get("https://steamcommunity.com/id/alaotach/?xml=1").mock(
        return_value=httpx.Response(
            200,
            text="<profile><steamID>alaotach</steamID><memberSince>May 1, 2020</memberSince><location>India</location></profile>",
        )
    )
    respx.get("https://pypi.org/user/alaotach/").mock(
        return_value=httpx.Response(
            200,
            text='<h1 class="author-profile__name">Aryan</h1><h3 class="package-snippet__title"><span class="package-snippet__name">austint</span></h3>',
        )
    )
    respx.get("https://medium.com/feed/@alaotach").mock(
        return_value=httpx.Response(
            200,
            text="<rss><channel><title>Stories by Aryan on Medium</title><description>AI notes</description><item><title>Post 1</title></item></channel></rss>",
        )
    )

    async with httpx.AsyncClient() as client:
        sc = await EXTRACTOR_REGISTRY["soundcloud"]("https://soundcloud.com/alaotach", "alaotach", client)
        yt = await EXTRACTOR_REGISTRY["youtube"]("https://www.youtube.com/@alaotach", "alaotach", client)
        tg = await EXTRACTOR_REGISTRY["telegram"]("https://t.me/mrmango42", "mrmango42", client)
        st = await EXTRACTOR_REGISTRY["steam"]("https://steamcommunity.com/id/alaotach", "alaotach", client)
        py = await EXTRACTOR_REGISTRY["pypi"]("https://pypi.org/user/alaotach/", "alaotach", client)
        md = await EXTRACTOR_REGISTRY["medium"]("https://medium.com/@alaotach", "alaotach", client)

    assert sc.followers == 42
    assert sc.post_count == {"tracks": 5}
    assert yt.display_name == "alaotach"
    assert yt.extra["subscribers_text"] == "1.2K subscribers"
    assert tg.followers == 1200
    assert st.joined_at == "May 1, 2020"
    assert py.post_count == {"packages": 1}
    assert md.post_count == {"posts": 1}


def test_ml_text_fallback_and_normalized_profile_schema():
    prof = ml_text_fallback_extract("Joined March 2021. Location: Bengaluru, India. 540 followers.")
    assert prof.joined_at == "March 2021"
    assert prof.followers == "540"
    assert prof.field_sources["joined_at"] == "ml"
    assert prof.field_confidence["joined_at"] == 0.45
    assert isinstance(prof, NormalizedProfile)
