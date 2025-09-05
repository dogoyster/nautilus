from __future__ import annotations

import sys
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from .config import Settings
from .binance_client import BinanceService


app = typer.Typer(add_completion=False, no_args_is_help=True, help="Simple Binance CLI (defaults to TESTNET)")
console = Console()


def _service() -> BinanceService:
    settings = Settings.from_env()
    if not settings.api_key or not settings.api_secret:
        console.print("[red]Missing BINANCE_API_KEY or BINANCE_API_SECRET in environment (.env).[/red]")
        raise typer.Exit(code=2)
    return BinanceService(settings)


@app.command()
def price(symbol: str = typer.Argument(..., help="Symbol like BTCUSDT")):
    """Get latest price for a symbol."""
    service = _service()
    p = service.get_price(symbol)
    console.print(f"[bold]{symbol.upper()}[/bold] price: [green]{p}[/green]")


@app.command()
def balances():
    """List non-zero balances with profit analysis."""
    service = _service()
    rows = service.get_balances()
    if not rows:
        console.print("No balances.")
        raise typer.Exit(code=0)
    
    t = Table(title="잔고 및 수익률 분석")
    t.add_column("자산", style="cyan")
    t.add_column("보유량", justify="right", style="green")
    t.add_column("잠김", justify="right", style="yellow")
    t.add_column("평균단가", justify="right", style="blue")
    t.add_column("현재가", justify="right", style="magenta")
    t.add_column("수익률", justify="right")
    t.add_column("수수료제외", justify="right")
    
    for b in rows:
        # 수익률 색상 결정
        profit_color = "green" if b.profit_rate > 0 else "red" if b.profit_rate < 0 else "white"
        no_fee_color = "green" if b.profit_rate_no_fee > 0 else "red" if b.profit_rate_no_fee < 0 else "white"
        
        # 평균단가가 0이면 "N/A" 표시
        avg_price_str = f"{b.avg_price:.4f}" if b.avg_price > 0 else "N/A"
        current_price_str = f"{b.current_price:.4f}" if b.current_price > 0 else "N/A"
        
        # 수익률 표시
        profit_str = f"{b.profit_rate:+.2f}%" if b.avg_price > 0 else "N/A"
        no_fee_str = f"{b.profit_rate_no_fee:+.2f}%" if b.avg_price > 0 else "N/A"
        
        t.add_row(
            b.asset,
            f"{b.free:.8f}",
            f"{b.locked:.8f}",
            avg_price_str,
            current_price_str,
            f"[{profit_color}]{profit_str}[/{profit_color}]",
            f"[{no_fee_color}]{no_fee_str}[/{no_fee_color}]"
        )
    
    console.print(t)


@app.command()
def buy(symbol: str, quantity: float = typer.Argument(..., help="Base asset quantity (e.g., 0.001 BTC)")):
    """Place a MARKET BUY order."""
    service = _service()
    order = service.market_buy(symbol, quantity)
    console.print(order)


@app.command()
def sell(symbol: str, quantity: float = typer.Argument(..., help="Base asset quantity (e.g., 0.001 BTC)")):
    """Place a MARKET SELL order."""
    service = _service()
    order = service.market_sell(symbol, quantity)
    console.print(order)


@app.command()
def orders(symbol: Optional[str] = typer.Argument(None, help="Optional symbol to filter")):
    """List open orders."""
    service = _service()
    data = service.get_open_orders(symbol)
    if not data:
        console.print("No open orders.")
        raise typer.Exit(code=0)
    console.print(data)


@app.command()
def cancel(symbol: str, order_id: int = typer.Argument(..., help="Order ID to cancel")):
    """Cancel an order by ID."""
    service = _service()
    res = service.cancel_order(symbol, order_id)
    console.print(res)


@app.command()
def limit_buy(symbol: str, quantity: float = typer.Argument(..., help="Base asset quantity"), 
              price: float = typer.Argument(..., help="Limit price")):
    """Place a LIMIT BUY order."""
    service = _service()
    order = service.limit_buy(symbol, quantity, price)
    console.print(f"[green]지정가 매수 주문 완료:[/green] {order}")


@app.command()
def limit_sell(symbol: str, quantity: float = typer.Argument(..., help="Base asset quantity"), 
               price: float = typer.Argument(..., help="Limit price")):
    """Place a LIMIT SELL order."""
    service = _service()
    order = service.limit_sell(symbol, quantity, price)
    console.print(f"[red]지정가 매도 주문 완료:[/red] {order}")


@app.command()
def stop_loss(symbol: str, quantity: float = typer.Argument(..., help="Base asset quantity"), 
              stop_price: float = typer.Argument(..., help="Stop loss price")):
    """Place a STOP LOSS order."""
    service = _service()
    order = service.stop_loss(symbol, quantity, stop_price)
    console.print(f"[yellow]손절매 주문 완료:[/yellow] {order}")


@app.command()
def take_profit(symbol: str, quantity: float = typer.Argument(..., help="Base asset quantity"), 
                price: float = typer.Argument(..., help="Take profit price")):
    """Place a TAKE PROFIT order."""
    service = _service()
    order = service.take_profit(symbol, quantity, price)
    console.print(f"[green]익절매 주문 완료:[/green] {order}")


if __name__ == "__main__":
    app()


