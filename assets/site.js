(() => {
  const synchronize = (group) => {
    const videos = Array.from(group.querySelectorAll("video"));
    if (videos.length !== 2) return;
    let updating = false;

    const setTime = (time) => {
      updating = true;
      videos.forEach((video) => {
        if (Number.isFinite(video.duration)) {
          video.currentTime = Math.min(time, Math.max(0, video.duration - 0.02));
        }
      });
      updating = false;
    };

    videos.forEach((video) => {
      video.addEventListener("play", () => {
        if (updating) return;
        videos.forEach((peer) => {
          if (peer !== video && peer.paused) peer.play().catch(() => {});
        });
      });
      video.addEventListener("pause", () => {
        if (updating) return;
        videos.forEach((peer) => {
          if (peer !== video && !peer.paused) peer.pause();
        });
      });
      video.addEventListener("seeking", () => {
        if (!updating) setTime(video.currentTime);
      });
    });

    window.setInterval(() => {
      if (videos.some((video) => video.paused || video.seeking)) return;
      const drift = videos[1].currentTime - videos[0].currentTime;
      if (Math.abs(drift) > 0.08) videos[1].currentTime = videos[0].currentTime;
    }, 250);
  };

  document
    .querySelectorAll("[data-sync-group]")
    .forEach((group) => synchronize(group));
})();
