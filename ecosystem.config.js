module.exports = {
  apps: [
    {
      name: "chinvex-tunnel",
      script: "cloudflared",
      args: "tunnel --protocol http2 run chinvex-gateway",
      autorestart: true,
      cron_restart: "0 */4 * * *",
      restart_delay: 5000
    },
    {
      name: "chinvex-gateway",
      script: "python",
      args: "-m chinvex.cli gateway serve --port 7778",
      autorestart: true,
      restart_delay: 2000
    },
  ]
}
